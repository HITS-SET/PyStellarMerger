import numpy as np

PI = 3.14159265358979323846

# cgs units
uCM = 1
uGR = 1
uSEC = 1
uERG = 1
uKELVIN = 1
uGS = 1
uAM = 1

# Derived units
uMSUN = 1.9892e33 * uGR
uRSUN = 6.9598e10 * uCM
uLSUN = 3.826e33 * uERG/uSEC
uC = 2.997924800e10 * uCM/uSEC
uG = 6.67259e-8 * uCM*uCM*uCM/uGR/uSEC
uE = 4.8032068e-10 * uAM
uH = 6.6260755e-27 * uCM*uCM*uGR/uSEC
uHBAR = uH / (2.0*PI)
uM_E = 9.1093897e-28 * uGR
uM_P = 1.6726231e-24 * uGR
uM_N = 1.674929e-24 * uGR
uM_U = 1.660540e-24 * uGR
uALPHA = 1.0/(uH*uC/(2*PI*uE*uE))
uSIGMA_TH = 6.6524616e-25 * uCM*uCM
uK = 1.380658e-16 * uERG/uKELVIN
uN_A = 6.0221367e+23
uSIGMA_RAD = 5.67051e-5 * uERG/uSEC/(uCM*uCM)/(uKELVIN*uKELVIN*uKELVIN*uKELVIN)
uA_RAD = 4.0*uSIGMA_RAD/uC
uEV = 1.602192e-12 * uERG
uKEV = 1.0e3 * uEV
uMEV = 1.0e6 * uEV

uHOUR = 3600.0 * uSEC
uDAY = 24.0 * uHOUR
uYR = 365.242 * uDAY

uAU = 1.49598e+13 * uCM
uPC = 3600.0*180.0/PI * uAU
uLY = uYR * uC
uKPC = 1.0e3 * uPC
uMPC = 1.0e6 * uPC

uKG = 1.0e-3 * uGR
uM = 100 * uCM
uKM = 1.0e3 * uM
uKMH = uKM / uHOUR
uW = 1.0e7 * uERG / uSEC
uJY = 1.0e-26 * uW / (uM*uM)
uMJY = 1.0e-3 * uJY
uDEG = PI / 180.0
uARCMIN = uDEG / 60.0
uARCSEC = uARCMIN / 60.0
