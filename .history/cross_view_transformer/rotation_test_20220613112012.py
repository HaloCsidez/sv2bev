from scipy.spatial.transform import Rotation as R
r = R.from_quat([0.5077241387638071,
-0.4973392230703816,
0.49837167536166627,
-0.4964832014373754])
matrix = r.as_matrix()
print("CAM_FRONT" + matrix)
euler = r.as_euler('zyx',degrees=True)
print(euler)
