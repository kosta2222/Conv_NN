import numpy as np
import traceback as t
import sys
import random
class CNN:
    def __init__(self):
        pass
    def init_net(self,l_r):
        self.F = 3
        self.F1 = 2
        self.W = 7
        self.d = 3
        self.S = 1
        self.P = 0
        # биас
        self.b0 = +1
        np.random.seed(42)
        # тестировочный подвыборочны слой
        self.X = np.random.normal(0, 1, (self.W, self.W, self.d))
        # окно для него
        self.W0 = np.random.normal(0, 1, (self.F, self.F, self.d))
        # окно для макс-пулинг
        self.W1 = np.random.normal(0, 1, (self.F1, self.F1, self.d))
        self.matrix_f_layer = self.set_io(5, 4)
        self.matrix_s_layer = self.set_io(4, 2)
        # неакивированное состояние первого слоя
        self.operated_val2 = None
        # неактивированное состояние второго слоя
        self.entered_val1 = None
        # тензор после свертки
        self.res_conv = None
        # его пропустили через активацию
        self.res_conv_act = None
        # активировали первый слой
        self.hidden1 = None
        # активировали второй слой
        self.hidden2 = None
        # тензор после макс-пулинга
        self.res_maxpooling = None
        # векторизировали тензор макс-пулинга
        self.signals_conv = None
        # коэффициент обучения
        self.l_r = l_r
    def create_train_set_CNN(self):
        l=[]
        for i in range(4):
            l.append(np.random.normal(0, 1, (self.W, self.W, self.d)))
        return l
    def create_train_set_FCNN(self):
        l=[]
        for i in range(4):
            l.append(np.random.normal(0,1,(5)))
        return l
    def create_answers(self):
        l=[]
        for i in range(4):
            l.append(np.random.randint(0,1,(2)))
        return l
    def rand_weight(self, out):
        return ((random.random() / 1 - 0.5) * (out ** -0.5))
    def set_io(self,inn, out):
        matrix = []
        row = [0] * inn
        for i in range(out):
            matrix.append(row)
        for row in range(len(matrix)):
            for elem in range(len(matrix[0])):
                matrix[row][elem] = self.rand_weight(out)
        return matrix
    def make_hidden(self,matrix,signals):
        hidden=[0]*(len(matrix))
        for row in range(len(matrix)):
            tmp_v=0
            for elem in range(len(matrix[0])):
               tmp_v+=matrix[row][elem]*signals[elem]
            hidden[row]=tmp_v
        return hidden
    def relu(self, val: float):
        if val < 0:
            return 0
        return val
    def derivate_relu(self, val: float):
        if val < 0:
            return 0.001
        return 1.0
    def conv(self, X, W0):
        W = len(X)
        F = len(W0)
        S = 1
        P = 0
        OutV = int((W - F + 2 * P) / S + 1)
        OutDepth = 1
        V1 = np.zeros((OutV, OutV, OutDepth))
        for downY in range(OutV):
            for acrossX in range(OutV):
                V1[downY, acrossX, 0] = np.sum(
                    X[S * downY:S * downY + F, S * acrossX:S * acrossX + F, :] * W0) + self.b0
        return V1.tolist()
    def conv_act(self, V1: np.ndarray) -> np.ndarray:
        V2 = np.zeros((len(V1), len(V1), len(V1[2])))
        for row in range(len(V1[0])):
            for elem in range(len(V1[1])):
                V2[row][elem][0] = self.relu(V1[row][elem][0])
        return V2.tolist()
    def maxpooling(self, X1: np.ndarray, W1: np.ndarray, S) -> np.ndarray:
        W = X1.shape[0]
        F1 = W1.shape[0]
        P = 0
        OutV = int((W - F1 + 2 * P) / S + 1)
        OutDepth = 1
        V3 = np.zeros((OutV, OutV, OutDepth))
        for downY in range(OutV):
            for acrossX in range(OutV):
                V3[downY, acrossX, 0] = np.max(X1[S * downY:S * downY + F1, S * acrossX:S * acrossX + F1, :])
        return V3
    def calc_out_gradients_FCN(self,matrix, targets,out_nn):
       errors=[0]*len(matrix)
       for row in range(len(matrix)):
           errors[row]=(targets[row]-out_nn[row])*self.derivate_relu(self.hidden2[row])
       return  errors
    def calc_hid_gradients_FCN(self,matrix,targets,entered_vals):
       errors=[0]*len(matrix[0])
       for elem in range(len(matrix[0])):
           errors[elem]=0
           for row in range(len(matrix)):
                    errors[elem]+= targets[row]* matrix[row][elem] *self.derivate_relu(entered_vals[elem])
       return errors
    def calc_hid_gradients_zero_FCN(self, matrix, targets):
        errors = [0] * len(matrix[0])
        for elem in range(len(matrix[0])):
            errors[elem] = 0
            for row in range(len(matrix)):
                errors[elem]+=targets[row]* matrix[row][elem]
        return errors
    def upd_matrix_FCN(self,matrix,errors,entered_vals):
       for row in range(len(matrix)):
           for elem in range(len(matrix[0])):
                matrix[row][elem]+=errors[elem]*self.l_r*entered_vals[elem]
       return matrix
    def feed_forward(self, X):
        print("i")
        # self.res_conv = self.conv(self.X, self.W0.tolist())
        # self.res_conv_act = self.conv_act(self.res_conv)
        # self.signals_conv = np.array([self.res_conv_act])
        # self.signals_conv=self.signals_conv.flatten().tolist()
        # # print("sign conv",self.res_conv_act)
        # print(self.matrix_f_layer)
        self.hidden1 = self.make_hidden( self.matrix_f_layer,X)
        print(self.matrix_f_layer)
        self.hidden2 = self.make_hidden( self.matrix_s_layer,self.hidden1)
        return self.hidden2
    def train_step(self,X,Y):
        # out_nn=self.feed_forward(X);
        print("in feed_forward matrix1",self.matrix_f_layer)
        self.hidden1 = self.make_hidden( self.matrix_f_layer,X)
        self.hidden2 = self.make_hidden( self.matrix_s_layer,self.hidden1)
        errs_out=self.calc_out_gradients_FCN(self.matrix_s_layer,Y,self.hidden2)
        print("in train step errs_out",errs_out)
        errs2=self.calc_hid_gradients_FCN(self.matrix_s_layer,errs_out,self.hidden1)
        errs1=self.calc_hid_gradients_zero_FCN(self.matrix_f_layer,errs2)
        print("in feed_forward matrix1",self.matrix_f_layer)
        print("in feed forward ers1 %s ers2 %s"%(str(errs1),str(errs2)))
        self.matrix_s_layer=self.upd_matrix_FCN(self.matrix_s_layer,errs2,self.hidden1)
        self.matrix_f_layer=self.upd_matrix_FCN(self.matrix_f_layer,errs1,X)
        print("in feed_forward matrix1",self.matrix_f_layer)
        print("in feed_forward matrix2",self.matrix_s_layer)
    def fit(self, n_epochs: int, l_r: float) -> None:
        self.init_net(l_r)
        ep = 0
        X=self.create_train_set_FCNN();
        Y=self.create_answers();
        while (ep < n_epochs):
                for i in range(len(X)):
                    show_mse:float = self.train_step(X[i].tolist(), Y[i].tolist())
                    if ep % 1 == 0:
                        print("Error mse:", show_mse)
                ep+=1
    # ==========================================
try:
    cnn = CNN()
    cnn.fit(15,0.07)
except Exception as e:
    # with open('log','w') as f:
    #  t.print_exc(file=f)
    t.print_exc(file=sys.stdout)
# =========================================
