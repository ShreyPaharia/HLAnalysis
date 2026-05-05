from __future__ import annotations

import argparse
import asyncio
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from .runner import run


def main() -> None:
    p = argparse.ArgumentParser(description="HLAnalysis market data recorder")
    p.add_argument("--config", type=Path, default=Path("config/symbols.yaml"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="If set, also write logs to this file with daily rotation (UTC, 14d retention).",
    )
    args = p.parse_args()

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            TimedRotatingFileHandler(
                args.log_file, when="midnight", backupCount=14, utc=True
            )
        )
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
    )

    asyncio.run(run(args.config, args.data_root))


if __name__ == "__main__":
    main()
