import numpy as np

def sort_model(model):
    m_prev = 0
    entropy = []
    dm = []

    for i in range(model.n_shells):
        entropy.append(model.buoyancy[i])
        dm.append(model.mass[i] - m_prev)
        m_prev = model.mass[i]

    # Sort the indices of 'entropy' such that the values they represent are sorted from smallest to largest
    index = np.argsort(entropy)

    m_cur = 0

    Mass = []
    entr = []
    m_mu = []
    id_s = []

    for i in index:
        m_cur += dm[i]
        Mass.append(m_cur)
        entr.append(entropy[i])
        m_mu.append(model.mean_mu[i])
        id_s.append(i)

    return np.array(Mass), np.array(entr), np.array(m_mu), np.array(id_s)



