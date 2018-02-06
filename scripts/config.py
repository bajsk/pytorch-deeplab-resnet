#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

root_dir = os.path.dirname(os.path.realpath(__file__)) + "/../"
sys.path.insert(0, root_dir)

class Config():
    root_path = root_dir
    image_path = os.path.join(root_dir, "images")
    model_path = os.path.join(root_dir, "models")
    data_path = os.path.join(root_dir, "data")
    snapshot_path = os.path.join(root_dir, "data", "snapshots")
    model_name = "inhand_robot"
    
