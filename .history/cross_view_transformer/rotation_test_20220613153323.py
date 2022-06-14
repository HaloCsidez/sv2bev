from scipy.spatial.transform import Rotation as R

# CAM_FRONT_LEFT
r_front_left = R.from_quat([0.6757265034669446,
        -0.6736266522251881,
        0.21214015046209478,
        -0.21122827103904068])
matrix = r_front_left.as_matrix()
print("CAM_FRONT_LEFT \n" , matrix)
euler = r_front_left.as_euler('zyx',degrees=True)
print(euler)

# CAM_FRONT
r_front = R.from_quat([0.4998015430569128,
        -0.5030316162024876,
        0.4997798114386805,
        -0.49737083824542755])
matrix = r_front.as_matrix()
print("CAM_FRONT\n", )
euler = r_front.as_euler('zyx',degrees=True)
print(euler)

# CAM_FRONT_RIGHT
r_front_right = R.from_quat([0.2060347966337182,
        -0.2026940577919598,
        0.6824507824531167,
        -0.6713610884174485])
matrix = r_front_right.as_matrix()
print("CAM_FRONT_RIGHT\n", )
euler = r_front_right.as_euler('zyx',degrees=True)
print(euler)

# CAM_BACK_LEFT
r_back_left = R.from_quat([0.6924185592174665,
        -0.7031619420114925,
        -0.11648342771943819,
        0.11203317912370753])
matrix = r_back_left.as_matrix()
print("CAM_BACK_LEFT\n", )
euler = r_back_left.as_euler('zyx',degrees=True)
print(euler)

# CAM_BACK
r = R.from_quat([0.5037872666382278,
        -0.49740249788611096,
        -0.4941850223835201,
        0.5045496097725578])
matrix = r.as_matrix()
print("CAM_BACK\n", )
euler = r.as_euler('zyx',degrees=True)
print(euler)

# CAM_BACK_RIGHT
r = R.from_quat([0.12280980120078765,
        -0.132400842670559,
        -0.7004305821388234,
        0.690496031265798])
matrix = r.as_matrix()
print("CAM_BACK_RIGHT\n", )
euler = r.as_euler('zyx',degrees=True)
print(euler)