"""
FL_correction_torch.py
======================
PyTorch-accelerated FL correction (TorchCore).

Key improvements over the numba-CUDA path:
  - No disk I/O between attenuation computation and MLEM (everything in-memory).
  - MLEM runs on all slices simultaneously via batched F.affine_grid + F.grid_sample,
    avoiding the explicit H-matrix (~3.4 GB per element in the numba path).

Rotation convention
-------------------
  rot3D(img, angle_deg) uses ndimage.rotate with positive = CCW.
  The numba kernel (kernel_generate_H_single_angle) samples the input at the
  positions obtained by applying R_CW(θ) to output coordinates, which is
  mathematically equivalent to displaying a CCW-rotated image.

  In PyTorch, the affine matrix for "image appears rotated CCW by φ" is:
      theta = [[cos φ, -sin φ, 0],
               [sin φ,  cos φ, 0]]
  This is used for _rotate_batch(imgs, +angle) = CCW rotation (forward projection)
  and _rotate_batch(imgs, -angle) = CW rotation (backprojection / denominator).

Forward / backproject at a single angle θ
------------------------------------------
  Forward:   X_rot = rotate_CCW(X, θ)          # bring X into the rotated frame
             I_pred = (X_rot * att_θ).sum(rows)  # project along rows
  Backward:  ratio_2d = ratio.expand(rows) * att_θ
             bp = rotate_CW(ratio_2d, θ)         # = rotate_CCW(ratio_2d, -θ)
  Denom:     denom += rotate_CW(att_θ, θ)       # H^T @ 1
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .FL_correction_core import Core
from .image_util import align_img


class TorchCore(Core):
    """Subclass of Core that replaces the disk-I/O + numba-CUDA MLEM path
    with an all-in-memory, PyTorch-batched equivalent."""

    # ------------------------------------------------------------------ #
    #  Internal rotation helper                                            #
    # ------------------------------------------------------------------ #

    def _rotate_batch(self, imgs, angle_deg, device):
        """Rotate a batch of 2D images by *angle_deg* degrees CCW.

        Parameters
        ----------
        imgs : (N, H, W) float32 tensor on *device*
        angle_deg : float  (positive = CCW, matching rot3D / ndimage.rotate)
        device : torch.device

        Returns
        -------
        (N, H, W) float32 tensor on *device*
        """
        phi = angle_deg * np.pi / 180.0
        cos_phi = float(np.cos(phi))
        sin_phi = float(np.sin(phi))
        # Affine matrix: maps output (x, y) -> input (x, y) for CCW-rotation
        # In PyTorch (x=col, y=row, both normalised to [-1,1]):
        #   x_in = cos(phi)*x_out - sin(phi)*y_out
        #   y_in = sin(phi)*x_out + cos(phi)*y_out
        theta_mat = torch.tensor(
            [[cos_phi, -sin_phi, 0.0],
             [sin_phi,  cos_phi, 0.0]],
            dtype=torch.float32, device=device,
        )                               # (2, 3)
        N, H, W = imgs.shape
        theta_batch = theta_mat.unsqueeze(0).expand(N, -1, -1)  # (N, 2, 3)
        grid = F.affine_grid(theta_batch, (N, 1, H, W), align_corners=False)
        out = F.grid_sample(
            imgs.unsqueeze(1), grid,
            mode='bilinear', padding_mode='zeros', align_corners=False,
        )
        return out.squeeze(1)   # (N, H, W)

    # ------------------------------------------------------------------ #
    #  In-memory attenuation + projection                                  #
    # ------------------------------------------------------------------ #

    def cal_and_save_atten_prj_torch(
        self, param, mu_probe, mu_fl, recon4D, rot_angles, ref_prj,
        fsave=None, align_flag=0, enable_scale=False, num_cpu=8,
    ):
        """In-memory replacement for cal_and_save_atten_prj.

        Returns
        -------
        atten_by_elem : dict  elem -> ndarray (n_ang, n_sli, n_row, n_col)
            Attenuation volumes per angle, kept in RAM.
        prj : ndarray  (n_elem, n_ang, n_sli, n_col)
            (Possibly alignment-shifted) observed projections.
        """
        s = recon4D.shape          # (n_elem, n_sli, n_row, n_col)
        elem_type = param['elem_type']
        Nelem     = len(elem_type)
        n_angle   = len(rot_angles)

        prj          = np.zeros([Nelem, n_angle, s[1], s[3]], dtype=np.float32)
        ref_prj_sum  = np.sum(ref_prj, axis=0)   # (n_ang, n_sli, n_col)

        atten_by_elem = {
            e: np.zeros((n_angle, s[1], s[2], s[3]), dtype=np.float32)
            for e in elem_type
        }

        for ang_id in range(n_angle):
            angle = rot_angles[ang_id]
            print(
                f'\r  [torch] computing attenuation at angle {angle:.1f}°'
                f'  {ang_id+1}/{n_angle}',
                end='', flush=True,
            )
            res = self.cal_atten_prj_at_angle(
                angle, recon4D, param, mu_probe, mu_fl,
                position_det='r', enable_scale=enable_scale, num_cpu=num_cpu,
            )
            if align_flag:
                _, r, c = align_img(res['prj_sum'], ref_prj_sum[ang_id])

            for i, elem in enumerate(elem_type):
                atten_by_elem[elem][ang_id] = res['atten'][elem].astype(np.float32)
                if align_flag:
                    prj[i, ang_id] = ndimage.shift(ref_prj[i, ang_id], [r, c])
                else:
                    prj[i, ang_id] = ref_prj[i, ang_id]

            # Optional on-disk write for debugging / compatibility
            if fsave is not None:
                for i, elem in enumerate(elem_type):
                    self.write_projection(
                        'm', elem, prj[i, ang_id],
                        angle, ang_id, fsave,
                    )
                    self.write_attenuation(
                        elem, res['atten'][elem],
                        angle, ang_id, fsave,
                    )

        print()   # newline after progress
        return atten_by_elem, prj

    # ------------------------------------------------------------------ #
    #  Batched MLEM (no explicit H matrix)                                 #
    # ------------------------------------------------------------------ #

    def mlem_torch(
        self, img3D, atten4D, I_obs, rot_angles, n_iter, device,
    ):
        """Batched MLEM for all slices simultaneously on *device*.

        Parameters
        ----------
        img3D   : (n_sli, n_row, n_col) ndarray – initial guess (e.g. ones)
        atten4D : (n_ang, n_sli, n_row, n_col) ndarray – attenuation per angle
        I_obs   : (n_sli, n_ang, n_col) ndarray – observed projections
        rot_angles : (n_ang,) array-like – angles in degrees
        n_iter  : int
        device  : torch.device

        Returns
        -------
        (n_sli, n_row, n_col) ndarray  corrected reconstruction
        """
        n_sli, n_row, n_col = img3D.shape
        n_ang = len(rot_angles)

        X = torch.from_numpy(img3D.clip(0)).float().to(device)   # (n_sli, n_row, n_col)
        A = torch.from_numpy(atten4D).float().to(device)         # (n_ang, n_sli, n_row, n_col)
        I = torch.from_numpy(I_obs).float().to(device)           # (n_sli, n_ang, n_col)
        I.clamp_(min=0)

        # ── Denominator: H^T @ 1  ──────────────────────────────────────
        # For each angle, backproject all-ones through att:
        #   denom_ang[sli, r, c] = rotate_CW(att[ang, sli], θ)[r, c]
        denom = torch.zeros_like(X)
        for ang_id in range(n_ang):
            angle = float(rot_angles[ang_id])
            A_ang = A[ang_id]                                     # (n_sli, n_row, n_col)
            # CW rotation = CCW by -angle
            denom += self._rotate_batch(A_ang, -angle, device)
        denom.clamp_(min=1e-6)

        # ── MLEM iterations  ───────────────────────────────────────────
        for _it in range(n_iter):
            numerator = torch.zeros_like(X)
            for ang_id in range(n_ang):
                angle = float(rot_angles[ang_id])
                A_ang = A[ang_id]                                 # (n_sli, n_row, n_col)

                # Forward: rotate X CCW by angle, multiply by att, sum rows
                X_rot  = self._rotate_batch(X, angle, device)    # (n_sli, n_row, n_col)
                I_pred = (X_rot * A_ang).sum(dim=1)               # (n_sli, n_col)
                I_pred.clamp_(min=1e-6)

                ratio  = I[:, ang_id, :] / I_pred                 # (n_sli, n_col)

                # Backproject: expand ratio over rows, mult by att, rotate CW
                ratio_2d = ratio.unsqueeze(1).expand(-1, n_row, -1) * A_ang
                numerator += self._rotate_batch(ratio_2d, -angle, device)

            X = (X * numerator / denom).clamp_(min=0)

        return X.cpu().numpy()

    # ------------------------------------------------------------------ #
    #  Top-level wrapper (one element)                                     #
    # ------------------------------------------------------------------ #

    def cuda_absorption_correction_torch(
        self, elem, ref_tomo, atten4D_elem, I_obs_elem,
        rot_angles, n_iter, device,
    ):
        """Run mlem_torch for a single element.

        Parameters
        ----------
        elem         : str
        ref_tomo     : (n_sli, n_row, n_col) ndarray – initial guess
        atten4D_elem : (n_ang, n_sli, n_row, n_col) ndarray
        I_obs_elem   : (n_sli, n_ang, n_col) ndarray
        rot_angles   : (n_ang,) degrees
        n_iter       : int
        device       : torch.device

        Returns
        -------
        (n_sli, n_row, n_col) ndarray
        """
        return self.mlem_torch(
            ref_tomo, atten4D_elem, I_obs_elem,
            rot_angles, n_iter, device,
        )
