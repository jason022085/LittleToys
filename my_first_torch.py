# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:41:20 2020

@author: USER
"""

from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)


import torch
torch.cuda.is_available()