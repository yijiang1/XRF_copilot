from workflow import XRFSim
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XRF Copilot"
    )
    parser.add_argument("-u", type=str, default="User", help="User name")
    parser.add_argument("-v", type=bool, default=False, help="Verbose mode")
    parser.add_argument("-llm", type=str, default="gpt-4o-mini", help="LLM model")

    args = parser.parse_args()

    workflow = XRFSim(llm=args.llm, user=args.u, verbose=args.v)
    result = workflow.run()
