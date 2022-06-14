from scipy.spatial.transform import Rotation as R
r = R.from_quat([0.7048620297871717,
-0.6907306801461466,
-0.11209091960167808,
0.11617345743327073])
matrix = r.as_matrix()
print("CAM_FRONT \n" , matrix)
euler = r.as_euler('zyx',degrees=True)
print(euler)
