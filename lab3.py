import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn.datasets
from sklearn.metrics import classification_report



labelStrings = []
with open("./iris.csv") as f:
    D = []
    L = []
    for line in f.readlines():
        fields = line.split(',')
        features = list(map(float, fields[0:4]))
        if not fields[4] in labelStrings:
            labelStrings.append(fields[4])
        L.append(labelStrings.index(fields[4]))
        D.append(features)

    D = np.array(D).T
    L = np.array(L)
    print(D)
    print(L)

    D0 = D[:, (L == 0)]
    D1 = D[:, (L == 1)]
    D2 = D[:, (L == 2)]

def vcol(x):
    return x.reshape(-1, 1)
def vrow(x):
    return x.reshape(1, -1)

u = D.mean(axis=1, keepdims=True)
DC = D - u
C = 1/D.shape[1]*DC@DC.T

def compute_PCA(D, L, m=2):
    u = D.mean(axis=1, keepdims=True)
    DC = D - u
    C = 1/D.shape[1]*DC@DC.T
    U, s, Vh = np.linalg.svd(C)
    P = U[:, 0:m]
    return P


P = compute_PCA(D, L, 2)
DP = np.dot(P.T, D)

U = D.mean(1, keepdims=True)
UC = []
NC = []

for i in range(0, 3):
    UC.append(D[:, (L == i)].mean(1))
    NC.append(np.count_nonzero(L == i))

UC = np.array(UC).T
print(UC[:, 0])

def plot_projected(DP, L):
    plt.figure()
    for l in np.unique(L):
        DPX = DP[0, (L == l)]
        DPY = DP[1, (L == l)]
        plt.scatter(DPX, DPY, alpha=0.5, label=f'Class {l}')
        plt.legend()


def plot_hist(DP, L, **kwargs):
    plt.figure()
    for l in np.unique(L):
        plt.hist(DP[0, (L == l)], alpha=0.5, label='fClass {l}', **kwargs)

plot_projected(DP, L)

UCP = np.dot(P.T, UC)
UP = np.dot(P.T, U)
plt.scatter(UP[0, 0], UP[1, 0], c='r')

plt.scatter(UCP[0, 0], UCP[1, 0], c='r')
plt.scatter(UCP[0, 1], UCP[1, 1], c='r')
plt.scatter(UCP[0, 2], UCP[1, 2], c='r')

plot_hist(DP, L)



def calc_SB_SW(D, L):
    NL = np.unique(L)
    print(L)
    U = D.mean(1, keepdims=True)
    UC = []
    NC = []

    for l in L:
        UC.append(D[:, (L == l)].mean(1))
        NC.append(np.count_nonzero(L == l))

    UC = np.array(UC).T

    SBsum = 0
    for i, l in enumerate(NL):
        SBsum += NC[i] * (UC[:, [i]] - U) @ (UC[:, [i]] - U).T

    SB = SBsum/D.shape[1]

    D_by_class = [D[:, (L == l)] for i, l in enumerate(NL)]
    SWsum = 0

    for i, l in enumerate(NL):
        SWsum += (D_by_class[i] - UC[:, [i]]) @ (D_by_class[i] - UC[:, [i]]).T

    SW = SWsum/D.shape[1]
    
    return (SB, SW)
SB, SW = calc_SB_SW(D, L)


def LDA(SB, SW, m=2):
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]
    return W
W = LDA(SB, SW)
DP = np.dot(W.T, D)

plot_projected(DP, L)


# LDA by joint diagonalization
def LDA_diag(D, L, m = 2):
    SB, SW = calc_SB_SW(D, L)
    U, s, _ = np.linalg.svd(SW)
    P1 = (U * 1.0/s**0.5) @ U.T

    SBT = P1.T @ SB @ P1
    U2, s2, _ = np.linalg.svd(SBT)
    P2 = U2[:, 0:m]
    return np.dot(P2.T, P).T

def apply_lda(U, D):
    return U.T @ D

def compute_lda_JointDiag(D, L, m):

    Sb, Sw = calc_SB_SW(D, L)

    U, s, _ = np.linalg.svd(Sw)
    P = U @ np.diag(1.0/(s**0.5)) @ U.T

    Sb2 = P @ Sb @ P.T
    U2, s2, _ = np.linalg.svd(Sb2)

    P2 = U2[:, 0:m]
    return P.T @ P2

ULDA = compute_lda_JointDiag(D, L, 2)
DP = apply_lda(ULDA, D)
plot_projected(DP, L)

plot_hist(DP, L)


def load_iris():
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

DIris, LIris = load_iris()
D = DIris[:, LIris != 0]
L = LIris[LIris != 0]

def center(D):
    return D - D.mean(1, keepdims=True)

def split_train(D, L, percentageTraining, seed=0):
    nTrain = int(D.shape[1] * percentageTraining)
    np.random.seed(seed)
    shuffledIndices = np.random.permutation(D.shape[1])

    DTR = D[:, shuffledIndices[0:nTrain]]
    LTR = L[shuffledIndices[0:nTrain]]

    DVAL = D[:, shuffledIndices[nTrain:]]
    LVAL = L[shuffledIndices[nTrain:]]

    return (DTR, LTR), (DVAL, LVAL)


def computeSwSb(D, L):
    '''
    Params:
    - D: Dataset features matrix, not ceCntered
    - L: Labels of the samples

    Returned Values:
    - Sw: Within-class scatter matrix
    - Sb: Between-class scatter matrix
    '''

    #find the unique labels for each class
    uniqueLabels = np.unique(L)

    #nc in the formula is computed as the number of samples of class c
    #separate data into classes
    DC = [D[:, L == label] for label in uniqueLabels]  #DC[0] -> samples of class 0, DC[1] -> samples of class 1 etc...

    #compute nc for each class
    #each element in DC has a shape which is (4, DC_i.shape[1]) (assuming samples are not equally distributed among all the classes which is true in 99% of cases...)
    #So for nc I just have to take DC_i.shape[1] for each i in DC
    nc = [DC_i.shape[1] for DC_i in DC]

    #Compute the mean as done before with PCA
    mu = D.mean(axis=1)
    mu = mu.reshape((mu.shape[0], 1))

    #Now compute the mean for each class
    muC = [DC[label].mean(axis=1) for label, labelName in enumerate(uniqueLabels)]
    muC = [mc.reshape((mc.shape[0], 1)) for mc in muC]

    Sb = 0  #between matrix initialization
    Sw = 0  #within  matrix initialization

    #iterate over all the classes to execute the summations to calculate the 2 matrices
    for label, labelName in enumerate(uniqueLabels):

        #1) FOR SB:
        #add up to the Sb (between) matrix
        diff = muC[label] - mu
        Sb += nc[label] * (diff @ (diff.T)) #nc * ((muC - mu) * (muC - mu).T)


        #2) FOR SW
        #add up to the Sw (within) matrix
        #for diff1 subtract the the class mean from the samples of each class, i.e center center the samples for each class 
        diff1 = DC[label] - muC[label]  #x_{c, i} - muC done by rows

        #SHORTCUT: compute the Sw matrix as a weighted sum of the covariance matrices of each class
        #so for each class:
        #Compute the Covariance Matrix C using DC = D - mu
        C_i = (diff1 @ diff1.T) / float(diff1.shape[1])  #Covariance matrix for class i

        #weighted sum of all the C_i
        Sw += nc[label] * C_i

    
    #at the end of the summations, just multiply by 1/N (N is the number of samples)

    Sb = Sb / D.shape[1]
    Sw = Sw / D.shape[1]

    #return both matrices
    return Sb, Sw

def classify_lda(D, L):
    (DTR, LTR), (DVAL, LVAL) = split_train(D, L, 2.0/3.0, seed=0)
    print("Training data shape: ", DTR.shape, LTR.shape)
    print("Validation data shape: ", DVAL.shape, LVAL.shape)
    ULDA = compute_lda_JointDiag(DTR, LTR, 1)
    DTR_lda = apply_lda(ULDA, DTR)
    DVAL_lda = apply_lda(ULDA, DVAL)

    # Check if the Virginica class samples are, on average, on the right of the Versicolor samples on the training set. If not, we reverse ULDA and re-apply the transformation.
    if DTR_lda[0, LTR==1].mean() > DTR_lda[0, LTR==2].mean():
        ULDA = -ULDA
        DTR_lda = apply_lda(ULDA, DTR)
        DVAL_lda = apply_lda(ULDA, DVAL)

    print("LDA W matrix", ULDA)


    plot_hist(DTR_lda, LTR, bins=5, density=True)
    plot_hist(DVAL_lda, LVAL, bins=5, density=True)

    # plot_projected(DTR_lda, LTR)

    threshold = (DTR_lda[0, LTR==1].mean() + DTR_lda[0, LTR==2].mean()) / 2.0 #
    print(threshold)

    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 2
    PVAL[DVAL_lda[0] < threshold] = 1
    print("LDA")
    print('Labels:     ', LVAL)
    print('Predictions:', PVAL)
    print('Number of erros:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.1f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))

    disagreements = np.count_nonzero(PVAL != LVAL)
    print(disagreements)


classify_lda(D, L)


def classify_PCA(D, L):
    (DTR, LTR), (DVAL, LVAL) = split_train(D, L, 2.0/3.0, seed=0)
    DTR = center(DTR)
    DVAL = center(DVAL)
    P = compute_PCA(DTR, LTR, 1)
    DTR_pca = np.dot(P.T, DTR)


    if DTR_pca[0, LTR==1].mean() > DTR_pca[0, LTR==2].mean():
        P = -P
        DTR_pca = np.dot(P.T, DTR)
        DVAL_pca = np.dot(P.T, DVAL)

    plot_hist(DTR_pca, LTR, bins=5, density=True)
    plot_hist(DVAL_pca, LVAL, bins=5, density=True)

    threshold = (DTR_pca[0, LTR==1].mean() + DTR_pca[0, LTR==2].mean()) / 2.0 #
    print(threshold)
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_pca[0] >= threshold] = 2
    PVAL[DVAL_pca[0] < threshold] = 1
    print("PCA")
    print('Labels:     ', LVAL)
    print('Predictions:', PVAL)
    print('Number of erros:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.1f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))
classify_PCA(D, L)

def singleAxis1DScatterPlot(x, y):
    plt.figure()
    labelColors = {1: "green", 2: "blue"}
    classLabels = {1: 'Iris-versicolor', 2: 'Iris-virginica'}


    for label in labelColors:
        xAxis = x[0, y == label]        #this are the real points (1-dimensional)
        yAxis = x[0, y == label]        #this is done to let the points stay on the 45 degrees axis to make them more visualizable 
        plt.scatter(xAxis, yAxis, color= labelColors[label], alpha=0.7,  label=f"{classLabels[label]}", edgecolor="black")

        #plt.label(featuresNames[i])
        #plt.ylabel(featuresNames[j])
        plt.legend()
        plt.title(f"Single Axis 1-D Scatter plot (45 degrees Axis chosen for better visualization), (DVAL, LVAL)")
        plt.xlabel("Projected Feature Value")
        plt.ylabel("Points on the 45 degrees axis (NOT real coordinates)")
    
    plt.show()

def jitter1DScatterPlot(x, y):
    #Adds jitter since the points are one dimensional so we have to distinguish them better
    labelColors = {1: "green", 2: "blue"}
    classLabels = {1: 'Iris-versicolor', 2: 'Iris-virginica'}
    
    #What does jitter do? It introduces a variation on the Y axis to better visualize the points
    #In fact, since the points are just 1-dimensional, they would stay just on the X axis
    #so jitter randomize their Y coordinates 
    np.random.seed(42)  
    jitter = np.random.normal(0, 0.05, size=x.shape[1])  
    
    for label in labelColors:
        xFeature = x[0, y == label]
        yJitter = jitter[y == label]  # Add a different randomized jitter for each class
        plt.scatter(xFeature, yJitter, color=labelColors[label], alpha=0.7, 
                    label=f"{classLabels[label]}", edgecolor="black")

    plt.xlabel("Projected Feature Value")
    plt.ylabel("Jitter (for visualization -- NOT real coordinates!)")
    plt.legend()
    plt.title("1D Scatter plot with jitter (DVAL, LVAL)")
    plt.show()

# PCA to 3 dimensions, LDA to 1 dimension
def classify_PCA_LDA(D, L, m1=3):
    (DTR, LTR), (DVAL, LVAL) = split_train(D, L, 2.0/3.0)
    P = compute_PCA(DTR, LTR, m1)
    DTR_pca = np.dot(P.T, DTR)
    DVAL_pca = np.dot(P.T, DVAL)


    if DTR_pca[0, LTR==1].mean() > DTR_pca[0, LTR==2].mean():
        P = -P
        DTR_pca = np.dot(P.T, DTR)
        DVAL_pca = np.dot(P.T, DVAL)

    # Plot after pca
    plot_hist(DTR_pca, LTR, bins=5, density=True)
    plot_hist(DVAL_pca, LVAL, bins=5, density=True)


    ULDA = compute_lda_JointDiag(DTR_pca, LTR, 1)
    DTR_lda = apply_lda(ULDA, DTR_pca)
    DVAL_lda = apply_lda(ULDA, DVAL_pca)

    if DTR_lda[0, LTR==1].mean() > DTR_lda[0, LTR==2].mean():
        ULDA = -ULDA
        DTR_lda = apply_lda(ULDA, DTR_pca)
        DVAL_lda = apply_lda(ULDA, DVAL_pca)


    plot_hist(DTR_lda, LTR, bins=5, density=True)
    plot_hist(DVAL_lda, LVAL, bins=5, density=True)

    singleAxis1DScatterPlot(DVAL_lda, LVAL)
    jitter1DScatterPlot(DVAL_lda, LVAL)

    # plot_projected(DTR_lda, LTR)

    #Build the model rule and apply it JUST on the VALIDATION SAMPLES

    #Since the projected samples have only one dimension, we can calculate the mean of the mean of the 2 classes simply like this (without a for loop):
    threshold = (DTR_lda[0, LTR==1].mean() + DTR_lda[0, LTR==2].mean()) / 2.0

    print(f"Selected threshold is: {threshold}")

    #Build an array to contain the predicted results, havinh shape equal to LVAL (34, )
    #At the begoinning it's preallocated with all zero
    #Later on it will store just 1 (class 1) or 2 (class 2)
    PVAL_model3 = np.zeros(shape=LVAL.shape, dtype=np.int32)

    #Classification rule:
    #- elements located at the right of the threshold -> classified as **class 2** (*Iris-virginica*)
    #- elements located at the left of the threshold -> classified as **class 1** (*Iris-versicolor*)

    PVAL_model3[DVAL_lda[0] >= threshold] = 2   #AT THE RIGHT OF THRESHOLD -> CLASS 2
    PVAL_model3[DVAL_lda[0] < threshold] = 1   #AT THE LEFT OF THRESHOLD -> CLASS 1
    plt.scatter([threshold], [0], c='r', s=100, marker='x', label='Threshold')
    jitter1DScatterPlot(DVAL_lda, PVAL_model3)

    #Confront the predictions to teh actual classes:

    error_count = np.count_nonzero(PVAL_model3 != LVAL)
    print(f"Number of wrong predictions: {error_count}")
    error_rate = np.mean(PVAL_model3 != LVAL)
    print(f"Error Rate: {error_rate:.2%}")
    print(classification_report(LVAL, PVAL_model3, digits=3))


classify_PCA_LDA(D, L, 2)


