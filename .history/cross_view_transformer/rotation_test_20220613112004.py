from scipy.spatial.transform import Rotation as R
r = R.from_quat([0.4998015430569128, -0.5030316162024876, 0.4997798114386805, -0.49737083824542755])
matrix = r.as_matrix()
print("CAM_FRONT" + matrix)
euler = r.as_euler('zyx',degrees=True)
print(euler)
