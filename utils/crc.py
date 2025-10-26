import struct

from unitree_hg.msg import LowCmd

from .singleton import Singleton

import ctypes
import os
import platform

class CRC(Singleton):
    def __init__(self):
        #4 bytes aligned, little-endian format.
        #size 1004
        self.__packFmtHGLowCmd = '<2B2x' + 'B3x5fI' * 35 + '5I'
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.platform = platform.system()
        if self.platform == "Linux":
            if platform.machine()=="x86_64":
                self.crc_lib = ctypes.CDLL(script_dir + '/lib/crc_amd64.so')
            elif platform.machine()=="aarch64":
                self.crc_lib = ctypes.CDLL(script_dir + '/lib/crc_aarch64.so')

            self.crc_lib.crc32_core.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32)
            self.crc_lib.crc32_core.restype = ctypes.c_uint32
    
    def Crc(self, msg):
        return self.__Crc32(self.__PackHGLowCmd(msg))


    def __PackHGLowCmd(self, cmd:LowCmd):
        origData = []
        origData.append(cmd.mode_pr)
        origData.append(cmd.mode_machine)

        for i in range(35):
            origData.append(cmd.motor_cmd[i].mode)
            origData.append(cmd.motor_cmd[i].q)
            origData.append(cmd.motor_cmd[i].dq)
            origData.append(cmd.motor_cmd[i].tau)
            origData.append(cmd.motor_cmd[i].kp)
            origData.append(cmd.motor_cmd[i].kd)
            origData.append(cmd.motor_cmd[i].reserve)

        origData.extend(cmd.reserve)
        origData.append(cmd.crc)

        return self.__Trans(struct.pack(self.__packFmtHGLowCmd, *origData))

    def __Trans(self, packData):
        calcData = []
        calcLen = ((len(packData)>>2)-1)

        for i in range(calcLen):
            d = ((packData[i*4+3] << 24) | (packData[i*4+2] << 16) | (packData[i*4+1] << 8) | (packData[i*4]))
            calcData.append(d)

        return calcData

    def _crc_py(self, data):
        bit = 0
        crc = 0xFFFFFFFF
        polynomial = 0x04c11db7

        for i in range(len(data)):
            bit = 1 << 31
            current = data[i]

            for b in range(32):
                if crc & 0x80000000:
                    crc = (crc << 1) & 0xFFFFFFFF
                    crc ^= polynomial
                else:
                    crc = (crc << 1) & 0xFFFFFFFF

                if current & bit:
                    crc ^= polynomial

                bit >>= 1
        
        return crc

    def _crc_ctypes(self, data):
        uint32_array = (ctypes.c_uint32 * len(data))(*data)
        length = len(data)
        crc=self.crc_lib.crc32_core(uint32_array, length)
        return crc

    def __Crc32(self, data):
        if self.platform == "Linux":
            return self._crc_ctypes(data)
        else:
            return self._crc_py(data)