from scipy.spatial.transform import Rotation as R
r = R.from_quat([0.20335173766558642,
-0.19146333228946724,
0.6785710044972951,
-0.6793609166212989])
matrix = r.as_matrix()
print("CAM_FRONT \n" , matrix)
euler = r.as_euler('zyx',degrees=True)
print(euler)
