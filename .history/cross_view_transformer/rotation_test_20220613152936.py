from scipy.spatial.transform import Rotation as R

r = R.from_quat([0.6757265034669446,
        -0.6736266522251881,
        0.21214015046209478,
        -0.21122827103904068])
matrix = r.as_matrix()
print("CAM_FRONT_LEFT \n" , matrix)
euler = r.as_euler('zyx',degrees=True)
print(euler)

r = R.from_quat([0.4998015430569128,
        -0.5030316162024876,
        0.4997798114386805,
        -0.49737083824542755])
matrix = r.as_matrix()
print("CAM_FRONT\n", )
euler = r.as_euler('zyx',degrees=True)
print(euler)

# CAM_FRONT_RIGHT
r = R.from_quat([0.6812088525125634,
-0.6687507165046241,
0.2101702448905517,
-0.21108161122114324])
matrix = r.as_matrix()
print("CAM_FRONT\n", )
euler = r.as_euler('zyx',degrees=True)
print(euler)

# CAM_BACK_LEFT
r = R.from_quat([0.6812088525125634,
-0.6687507165046241,
0.2101702448905517,
-0.21108161122114324])
matrix = r.as_matrix()
print("CAM_FRONT\n", )
euler = r.as_euler('zyx',degrees=True)
print(euler)

# CAM_BACK
r = R.from_quat([0.6812088525125634,
-0.6687507165046241,
0.2101702448905517,
-0.21108161122114324])
matrix = r.as_matrix()
print("CAM_FRONT\n", )
euler = r.as_euler('zyx',degrees=True)
print(euler)

# CAM_BACK_RIGHT
r = R.from_quat([0.6812088525125634,
-0.6687507165046241,
0.2101702448905517,
-0.21108161122114324])
matrix = r.as_matrix()
print("CAM_FRONT\n", )
euler = r.as_euler('zyx',degrees=True)
print(euler)