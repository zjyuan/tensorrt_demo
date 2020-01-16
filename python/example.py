from ctypes import cdll
import os

p = os.getcwd() + '/libinference.so'
i = cdll.LoadLibrary(p)
i.classify("demo.jpg".encode("ascii"))
i.detect("demo1.jpg".encode('ascii'))