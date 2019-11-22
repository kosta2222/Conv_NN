matrix=[[1,2,4,5],
        [6,7,8,9]] # матрица (2,4)

import  numpy as np
np.random.seed(42)
matr=np.random.randint(0,100,(4,4))
kern=np.random.randint(0,100,(2,2))
print("kern:\n",kern)
print("matr:\n",matr)
kern=kern.tolist()
matr_as_ribbon=matr.flatten().tolist()
def conv(matr_as_ribbon_one_channel, kernel_as_matr_one_channel):
    matr_as_ribbon_height=4
    matr_as_ribbon_width=4
    kernel_as_matr_one_channel_width=2
    S=1 # шаг
    V=((matr_as_ribbon_width - kernel_as_matr_one_channel_width) // S + 1) # выходной обьем(ширина) выходной карты признаков
    # выходная карта признаков квадратная
    result_conv_one_channel_as_ribbon=[0]*(V**2)
    n=-1
    for i in range(matr_as_ribbon_height-1):
        for j in range(matr_as_ribbon_width-1):
            result=0
            n += S
            for a in range(kernel_as_matr_one_channel_width):
                for b in range(kernel_as_matr_one_channel_width):
                    Y=i*S+a
                    X=j*S+b
                    print(matr_as_ribbon_one_channel[matr_as_ribbon_width*Y+X])
                    result+=matr_as_ribbon_one_channel[matr_as_ribbon_width*Y+X]*kernel_as_matr_one_channel[a][b]
                    result_conv_one_channel_as_ribbon[n]=result
    return result_conv_one_channel_as_ribbon

print(conv(matr_as_ribbon,kern))
