"""Strategy parameters from config/strategy.yaml."""

import yaml

from db import ROOT

STRATEGY = yaml.safe_load(open(ROOT / "config" / "strategy.yaml"))
