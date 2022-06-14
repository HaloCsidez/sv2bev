from scipy.spatial.transform import Rotation as R
r = R.from_quat([0.13819187705364147,
-0.13796718183628456,
-0.6893329941542625,
0.697630335509333])
matrix = r.as_matrix()
print("CAM_FRONT \n" , matrix)
euler = r.as_euler('zyx',degrees=True)
print(euler)
