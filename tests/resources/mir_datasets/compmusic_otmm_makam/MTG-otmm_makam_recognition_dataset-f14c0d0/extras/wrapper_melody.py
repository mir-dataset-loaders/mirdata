# -*- coding: utf-8 -*-
from morty.extras.pitch import Pitch
import os

folder = os.path.join('..', 'data')

p = Pitch()
p.extract(folder)
