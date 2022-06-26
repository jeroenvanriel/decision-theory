def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w


print('dqn1')
h, w = 84, 84
print(h,w)
h, w = conv_output_shape((h,w), kernel_size=8, stride=4)
print(h,w)
h, w = conv_output_shape((h,w), kernel_size=4, stride=2)
print(h,w)
h, w = conv_output_shape((h,w), kernel_size=3, stride=1)
print(h,w)

print('dqn3')
h, w = 84, 84
print(h,w)
h, w = conv_output_shape((h,w), kernel_size=8)
print(h,w)
h, w = conv_output_shape((h / 4, w / 4), kernel_size=4)
print(h,w)
h, w = conv_output_shape((h / 2, w / 2), kernel_size=3)
print(h/2,w/2)
