import numpy as np
import torch as tc
import torch.nn as nn
from .util import rotate

tc.set_default_dtype(tc.float32)


class PPM(nn.Module):

    def __init__(self, dev, model_self_absorption, lac, grid_concentration, p, n_element, n_lines,
                 mu_fl, detected_fl_unit_concentration, n_line_group_each_element,
                 sample_height_n, minibatch_size, sample_size_n, sample_size_cm,
                 probe_energy, incident_probe_intensity, model_probe_attenuation, mu_probe,
                 theta, signal_attenuation_factor,
                 n_det, P_minibatch, det_dia_cm, det_from_sample_cm, det_solid_angle_ratio,
                 debug=False):
        super(PPM, self).__init__()
        self.dev = dev
        self.model_self_absorption = model_self_absorption
        self.lac = lac
        self.grid_concentration = grid_concentration
        self.p = p
        self.n_element = n_element
        self.n_lines = n_lines

        self.mu_fl = mu_fl.to(self.dev)
        self.detected_fl_unit_concentration = detected_fl_unit_concentration.to(self.dev)
        self.n_line_group_each_element = n_line_group_each_element.to(self.dev)

        self.sample_height_n = sample_height_n
        self.minibatch_size = minibatch_size
        self.sample_size_n = sample_size_n
        self.sample_size_cm = sample_size_cm
        self.dia_len_n = int(1.2 * (self.sample_height_n**2 + self.sample_size_n**2 + self.sample_size_n**2)**0.5)
        self.n_voxel_minibatch = self.minibatch_size * self.sample_size_n
        self.n_voxel = self.sample_height_n * self.sample_size_n**2

        self.xp = self.init_xp()
        self.probe_energy = probe_energy
        self.incident_probe_intensity = incident_probe_intensity
        self.model_probe_attenuation = model_probe_attenuation
        self.mu_probe = mu_probe
        self.probe_before_attenuation_flat = self.init_probe()

        self.theta = theta
        self.signal_attenuation_factor = signal_attenuation_factor

        self.n_det = n_det
        self.P_minibatch = P_minibatch
        self.det_dia_cm = det_dia_cm
        self.det_from_sample_cm = det_from_sample_cm
        self.SA_theta = self.init_SA_theta(debug)
        self.det_solid_angle_ratio = det_solid_angle_ratio

        self.skip_rotation = (theta == 0)
        if self.skip_rotation and debug:
            print("PPM model: Theta is 0, will skip rotation operations")

    def init_xp(self):
        return nn.Parameter(self.grid_concentration[:, self.minibatch_size * self.p // self.sample_size_n : self.minibatch_size*(self.p+1) // self.sample_size_n, :, :])

    def init_SA_theta(self, debug=False):
        if self.model_self_absorption == True:
            voxel_idx_offset = self.p * self.n_voxel_minibatch

            if debug:
                print(f"init_SA_theta debug:")
                print(f"voxel_idx_offset: {voxel_idx_offset}")
                print(f"n_voxel_minibatch: {self.n_voxel_minibatch}")
                print(f"P_minibatch shape: {self.P_minibatch.shape}")

            try:
                safe_indices = []
                for m in range(self.n_det):
                    idx1 = tc.clamp(self.P_minibatch[m,0] - voxel_idx_offset, 0, self.n_voxel_minibatch-1).to(dtype=tc.long)
                    idx2 = tc.clamp(self.P_minibatch[m,1], 0, self.n_voxel-1).to(dtype=tc.long)

                    if debug:
                        if tc.any((self.P_minibatch[m,0] - voxel_idx_offset) < 0) or tc.any((self.P_minibatch[m,0] - voxel_idx_offset) >= self.n_voxel_minibatch):
                            print(f"Warning: Some indices in P_minibatch[{m},0] - offset are out of bounds")

                        if tc.any(self.P_minibatch[m,1] < 0) or tc.any(self.P_minibatch[m,1] >= self.n_voxel):
                            print(f"Warning: Some indices in P_minibatch[{m},1] are out of bounds")

                    lac_values = self.lac[:,:, idx1, idx2]
                    att_exponent_m = lac_values * self.P_minibatch[m,2].repeat(self.n_element, self.n_lines, 1)
                    safe_indices.append(att_exponent_m)

                att_exponent = tc.stack(safe_indices)
                att_exponent_voxel_sum = tc.sum(att_exponent.view(self.n_det, self.n_element, self.n_lines, self.n_voxel_minibatch, self.dia_len_n), axis=-1)
                SA_theta = tc.mean(tc.exp(-tc.sum(att_exponent_voxel_sum, axis=1)), axis=0)

            except Exception as e:
                if debug:
                    print(f"Error in init_SA_theta: {str(e)}")
                    print("Falling back to no self-absorption")
                SA_theta = tc.ones((self.n_lines, self.n_voxel_minibatch), device=self.dev)
        else:
            SA_theta = 1

        return SA_theta

    def init_probe(self):
        probe_before_attenuation = self.incident_probe_intensity * tc.ones(self.minibatch_size, self.sample_size_n, device = self.dev)
        return probe_before_attenuation.view(self.n_voxel_minibatch)

    def forward(self):
        concentration_map_minibatch = self.xp

        if not self.skip_rotation:
            concentration_map_minibatch_rot = rotate(concentration_map_minibatch, self.theta, self.dev)
            concentration_map_minibatch_rot = tc.reshape(concentration_map_minibatch_rot, (self.n_element, self.minibatch_size, self.sample_size_n))
        else:
            concentration_map_minibatch_rot = concentration_map_minibatch.view(self.n_element, self.minibatch_size, self.sample_size_n)

        att_exponent_acc_map = tc.zeros((self.minibatch_size, self.sample_size_n+1), device=self.dev)

        fl_map_tot_flat_theta = tc.zeros((self.n_lines, self.n_voxel_minibatch), device=self.dev)
        concentration_map_minibatch_rot_flat = concentration_map_minibatch_rot.view(self.n_element, self.n_voxel_minibatch)
        line_idx = 0
        for j in range(self.n_element):
            if self.model_probe_attenuation == True:
                lac_single = concentration_map_minibatch_rot[j] * self.mu_probe[j]
                lac_acc = tc.cumsum(lac_single, axis=1)
                lac_acc = tc.cat((tc.zeros((self.minibatch_size, 1), device=self.dev), lac_acc), dim = 1)
                att_exponent_acc = lac_acc * (self.sample_size_cm / self.sample_size_n)
                att_exponent_acc_map += att_exponent_acc
            else:
                att_exponent_acc_map = tc.zeros(self.minibatch_size, self.sample_size_n+1).to(self.dev)

            fl_unit = self.detected_fl_unit_concentration[line_idx:line_idx + self.n_line_group_each_element[j]]
            fl_map = tc.stack([concentration_map_minibatch_rot_flat[j] * fl_unit_single_line for fl_unit_single_line in fl_unit])
            fl_map_tot_flat_theta[line_idx:line_idx + self.n_line_group_each_element[j],:] = fl_map
            line_idx = line_idx + len(fl_unit)

        attenuation_map_theta_flat = tc.exp(-(att_exponent_acc_map[:,:-1])).view(self.n_voxel_minibatch)
        transmission_att_exponent_theta = att_exponent_acc_map[:,-1]

        probe_after_attenuation_theta = self.probe_before_attenuation_flat * attenuation_map_theta_flat
        fl_signal_SA_theta = tc.unsqueeze(probe_after_attenuation_theta, dim=0) * fl_map_tot_flat_theta * self.SA_theta
        fl_signal_SA_theta = fl_signal_SA_theta.view(self.n_lines, self.minibatch_size, self.sample_size_n)
        fl_signal_SA_theta = tc.sum(fl_signal_SA_theta, axis=-1)

        fl_signal_SA_theta = fl_signal_SA_theta * self.det_solid_angle_ratio * self.signal_attenuation_factor

        output1 = fl_signal_SA_theta
        output2 = transmission_att_exponent_theta

        return output1, output2
