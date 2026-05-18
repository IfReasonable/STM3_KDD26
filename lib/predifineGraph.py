import numpy as np
import scipy.sparse as sp
import pickle
import pandas as pd
import os

def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data



def calculate_scaled_laplacian(adj):
    n = adj.shape[0]
    d = np.sum(adj, axis=1)  # D
    lap = np.diag(d) - adj     # L=D-A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                lap[i, j] /= np.sqrt(d[i] * d[j])
    lap[np.isinf(lap)] = 0
    lap[np.isnan(lap)] = 0
    lam = np.linalg.eigvals(lap).max().real
    return 2 * lap / lam - np.eye(n)



def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    try:
        W = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    # check whether W is a 0/1 matrix.
    if set(np.unique(W)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        n = W.shape[0]
        W = W / 10000.
        W2, WMASK = W * W, np.ones([n, n]) - np.identity(n)
        # refer to Eq.10
        A = np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * WMASK
        return A
    else:
        return W


def first_approx(W, n):
    A = W + np.identity(n)
    d = np.sum(A, axis=1)
    sinvD = np.sqrt(np.mat(np.diag(d)).I)
    # refer to Eq.5
    return np.mat(np.identity(n) + sinvD * A * sinvD)

def get_normalized_adj(A):
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def idEncode(x, y, col):
    return x * col + y

def constructGraph(row, col):
    mx = [-1, 0, 1, 0, -1, -1, 1, 1, 0]
    my = [0, -1, 0, 1, -1, 1, -1, 1, 0]

    areaNum = row * col

    def illegal(x, y):
        return x < 0 or y < 0 or x >= row or y >= col

    W = np.zeros((areaNum, areaNum))
    for i in range(row):
        for j in range(col):
            n1 = idEncode(i, j, col)
            for k in range(len(mx)):
                temx = i + mx[k]
                temy = j + my[k]
                if illegal(temx, temy):
                    continue
                n2 = idEncode(temx, temy, col)
                W[n1, n2] = 1
    return W


def my_get_A(DATASET, num_nodes):
    filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    if DATASET == 'METR_LA':
        adj_file_path = os.path.join(filepath, 'METR_LA/adj_mx.pkl')
        sensor_ids, sensor_id_to_ind, A = load_pickle(pickle_file=adj_file_path)
    elif DATASET == 'PEMSD4':
        adj_file_path = os.path.join(filepath, 'PeMS04/PEMS04.csv')
        A, Distance = get_adjacency_matrix(
            distance_df_filename=adj_file_path,
            num_of_vertices=num_nodes)
    elif DATASET == 'PEMSD8':
        adj_file_path = os.path.join(filepath, 'PEMS08/PEMS08.csv')
        A, Distance = get_adjacency_matrix(
            distance_df_filename=adj_file_path,
            num_of_vertices=num_nodes)
    elif DATASET in ['Milan_internet', 'Milan_call', 'Milan_sms']:
        adj_file_path = os.path.join(filepath, 'Milan/adj_matrix.csv')
        A = pd.read_csv(adj_file_path, header=None).values
    elif DATASET in ['KnowAir', 'ETTh1', 'Electricity']:
        A = np.eye(num_nodes)
    elif DATASET == "NREL":
        adj_file_path = os.path.join(filepath, 'NREL/nerl_A.npy')
        A = np.load(adj_file_path)
    else:
        raise NotImplementedError(f'ERROR: dataset not supported: {DATASET}')
    
    return A
 