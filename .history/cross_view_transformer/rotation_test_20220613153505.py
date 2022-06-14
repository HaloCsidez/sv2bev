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
print("CAM_FRONT\n", matrix)
euler = r_front.as_euler('zyx',degrees=True)
print(euler)

# CAM_FRONT_RIGHT
r_front_right = R.from_quat([0.2060347966337182,
        -0.2026940577919598,
        0.6824507824531167,
        -0.6713610884174485])
matrix = r_front_right.as_matrix()
print("CAM_FRONT_RIGHT\n", matrix)
euler = r_front_right.as_euler('zyx',degrees=True)
print(euler)

# CAM_BACK_LEFT
r_back_left = R.from_quat([0.6924185592174665,
        -0.7031619420114925,
        -0.11648342771943819,
        0.11203317912370753])
matrix = r_back_left.as_matrix()
print("CAM_BACK_LEFT\n", matrix)
euler = r_back_left.as_euler('zyx',degrees=True)
print(euler)

# CAM_BACK
r_back = R.from_quat([0.5037872666382278,
        -0.49740249788611096,
        -0.4941850223835201,
        0.5045496097725578])
matrix = r_back.as_matrix()
print("CAM_BACK\n", matrix)
euler = r_back.as_euler('zyx',degrees=True)
print(euler)

# CAM_BACK_RIGHT
r_back_right = R.from_quat([0.12280980120078765,
        -0.132400842670559,
        -0.7004305821388234,
        0.690496031265798])
matrix = r_back_right.as_matrix()
print("CAM_BACK_RIGHT\n", matrix)
euler = r_back_right.as_euler('zyx',degrees=True)
print(euler)


# RESULT
# CAM_FRONT_LEFT 
#  [[ 2.44737995e-03 -8.20754770e-01  5.71275430e-01]
#  [-9.99994759e-01 -3.21950185e-03 -3.41436671e-04]
#  [ 2.11945808e-03 -5.71271601e-01 -8.20758348e-01]]
# [ 89.82915223  34.83921336 179.97616487]
# CAM_FRONT
#  [[-5.64133364e-03 -5.68014846e-03  9.99967955e-01]
#  [-9.99983763e-01  8.37115272e-04 -5.63666773e-03]
#  [-8.05071338e-04 -9.99983517e-01 -5.68477868e-03]]
# [134.80356665  89.54131123 135.24347929]
# CAM_FRONT_RIGHT
#  [[-1.36479031e-02  8.32817742e-01  5.53379023e-01]
#  [-9.99865858e-01 -1.63788158e-02 -9.94603768e-06]
#  [ 9.05540984e-03 -5.53304927e-01  8.32929563e-01]]
# [-9.09388576e+01  3.35991383e+01  6.84170676e-04]
# CAM_BACK_LEFT
#  [[-0.01601021 -0.94766474 -0.31886551]
#  [-0.99986478  0.0139763   0.00866572]
#  [-0.00375564  0.31896113 -0.94776036]]
# [  90.96788478  -18.59432961 -179.47613821]
# CAM_BACK
#  [[ 0.01674384 -0.00248837 -0.99985672]
#  [-0.99985181  0.00395911 -0.01675361]
#  [ 0.00400023  0.99998907 -0.00242171]]
# [  8.45308735 -89.03006613  98.22505222]
# CAM_BACK_RIGHT
#  [[-0.01626597  0.93476883 -0.35488399]
#  [-0.99980932 -0.0113705   0.01587584]
#  [ 0.01080503  0.35507456  0.93477554]]
# [-90.9969066  -20.7863348   -0.97299408]