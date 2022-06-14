from scipy.spatial.transform import Rotation as R
r = R.from_quat([0.5067997344989889,
-0.4977567019405021,
-0.4987849934090844,
0.496594225837321])
matrix = r.as_matrix()
print("CAM_FRONT \n" , matrix)
euler = r.as_euler('zyx',degrees=True)
print(euler)
