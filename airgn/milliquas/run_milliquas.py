from pathlib import Path
from ampel.cli.JobCommand import JobCommand

JOB_FILE = Path(__file__).parent / "milliquas_ampel_job.yml"
CONFIG_FILE = Path(__file__).parent.parent / "ampel_config.yml"

cmd = JobCommand()
parser = cmd.get_parser()
args = vars(
    parser.parse_args(
        [
            "--schema",
            str(JOB_FILE),
            "--config",
            str(CONFIG_FILE),
            "--task",
            "0",
        ]
    )
)
cmd.run(args, unknown_args=())
