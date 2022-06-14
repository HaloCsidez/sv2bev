from scipy.spatial.transform import Rotation as R

r = R.from_quat([0.6812088525125634,
-0.6687507165046241,
0.2101702448905517,
-0.21108161122114324])
matrix = r.as_matrix()
print("CAM_FRONT_LEFT \n" , matrix)
euler = r.as_euler('zyx',degrees=True)
print(euler)

r = R.from_quat([0.6812088525125634,
-0.6687507165046241,
0.2101702448905517,
-0.21108161122114324])
matrix = r.as_matrix()
print("CAM_FRONT\n", )
euler = r.as_euler('zyx',degrees=True)
print(euler)