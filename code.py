import skimage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.seterr(all='ignore') 

def within_class_variance(img, t):
    freq = np.zeros(256)
    for i in range(256):
        freq[i] = np.sum(img==i)
    m, n = img.shape
    p = freq/(m*n)
    # Class probabilities
    w0 = np.sum(p[:t+1])
    w1 = np.sum(p[t+1:])

    if w0==0 or w1==0:
        return np.inf
    
    # Class mean
    u0 = np.sum(np.arange(t+1)*p[:t+1])/w0
    u1 = np.sum(np.arange(t+1,256)*p[t+1:])/w1
    # Class variance
    s0 = np.sum(((np.arange(t+1)-u0)**2)*p[:t+1]) / w0
    s1 = np.sum(((np.arange(t+1, 256)-u1)**2)*p[t+1:]) / w1

    return w0*s0 + w1*s1

def between_class_variance(img, t): 
    freq = np.zeros(256)
    for i in range(256):
        freq[i] = np.sum(img==i)
    m, n = img.shape
    p = freq/(m*n)
    # Class probabilities
    w0 = np.sum(p[:t+1])
    w1 = np.sum(p[t+1:])
    # image mean
    uT = np.sum(np.arange(256)*p)
    if w0==0 or w1==0:
        return -np.inf
    # Mean value of intensity of a pixel belonging to a class
    u0 = np.sum(np.arange(t+1)*p[:t+1])/w0
    u1 = np.sum(np.arange(t+1,256)*p[t+1:])/w1
    # Between class variance
    return w0*(u0-uT)**2 + w1*(u1-uT)**2 

def adaptive_binarization(img, block_size):
    M, N = img.shape
    m, n = M//block_size, N//block_size
    v_s, h_s = 0, 0
    binarized_image = np.zeros((M, N))

    def between_class_variance(t, p):
        # Class probabilities
        w0 = np.sum(p[:t+1])
        w1 = np.sum(p[t+1:])
        # image mean
        uT = np.sum(np.arange(256)*p)
        if w0==0 or w1==0:
            return -np.inf
        # Mean value of intensity of a pixel belonging to a class
        u0 = np.sum(np.arange(t+1)*p[:t+1])/w0
        u1 = np.sum(np.arange(t+1,256)*p[t+1:])/w1
        # Between class variance
        return w0*(u0-uT)**2 + w1*(u1-uT)**2 

    while v_s < block_size:
        freq = np.zeros(256)
        img_plot = img[h_s*m:h_s*m+n, v_s*n:v_s*n+m]
        for i in range(256):
            freq[i] = np.sum(img_plot==i)
        p = freq / (m*n)
        between_class_variance_t = np.zeros(256)
        for t in range(256):
            between_class_variance_t[t] = between_class_variance(t, p)
        t = np.argmax(between_class_variance_t)
        # print(t)
        for i in range(m):
            for j in range(n):
                if img[h_s*m+i, v_s*n+j] > t:
                    binarized_image[h_s*m+i,v_s*n+j] = 1
                else:
                    binarized_image[h_s*m+i,v_s*n+j] = 0
        h_s += 1
        if h_s == block_size:
            h_s = 0
            v_s += 1
    return binarized_image

def connected_components(img):
    M, N = img.shape
    k = 0
    connected = np.ones_like(img)*np.inf
    for i in range(M):
        for j in range(N):
            if img[i, j] == 0:
                r1, c1 = max(0, i-1), max(0, j-1)
                c2 = min(N-1, j+1)
                c = np.min([connected[r1, c1], connected[i, c1], connected[r1, j], connected[r1, c2]])
                flag = False
                if r1!=i and c1!=j:
                    if img[r1, c1] == 0: 
                        connected[r1, c1] = c
                        flag = True
                    if img[i, c1] == 0:
                        connected[i, c1] = c
                        flag = True
                    if img[r1, j] == 0:
                        if connected[r1, c1] != np.inf:
                            connected[connected==connected[r1, j]] = c 
                        connected[r1, j] = c
                        flag = True
                    if img[r1, c2] == 0:
                        if connected[r1, c1] != np.inf:
                            connected[connected==connected[r1, c2]] = c 
                        connected[r1, c2] = c
                        flag = True
                if flag:
                    connected[i, j] = c
                if not flag:
                    k += 1
                    connected[i, j] = k

    # Removing punctuations
    unique = np.unique(connected)
    count = {}
    for i in unique:
        if i!=np.inf:
            c = int(np.sum(connected==i))
            count[int(i)] = int(np.sum(connected==i))
    df = pd.DataFrame(count.values())
    Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
    IQR = Q3 - Q1
    ans = 0
    for i in count:
        if count[i] > (Q1-IQR).values[0]:
            ans += 1
    return ans, connected

if __name__ == "__main__":
    img = skimage.io.imread("./images/coins.png")

    # Question 1
    freq = np.zeros(256)
    for i in range(256):
        freq[i] = np.sum(img==i)

    plt.plot(freq)
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.savefig("./plots/1.png")
    plt.close()

    m, n = img.shape
    p = freq/(m*n)
    average_intensity = round(np.sum(np.arange(256)*p), 2)
    print(f'Average intensity = {average_intensity}')
    assert average_intensity == round(np.mean(img), 2)
    print("Verified the average intensity with actual intensity")

    # Question 2
    t1, t2 = None, None

    within_class_variance_t = np.ones(256)*np.inf
    between_class_variance_t = np.ones(256)*(-1)*np.inf
    for t in range(0, 256):
        within_class_variance_t[t] = within_class_variance(img, t)
        between_class_variance_t[t] = between_class_variance(img, t)

    image_variance = within_class_variance_t + between_class_variance_t

    t1 = np.argmin(within_class_variance_t)
    t2 = np.argmax(between_class_variance_t)

    print(f't that minimizes within class variance = {t1}')
    print(f't that maximizes between class variance = {t2}')

    plt.plot(within_class_variance_t, label="sw^2")
    plt.plot(between_class_variance_t, label="sb^2")
    plt.plot(image_variance, label="st^2")
    plt.axvline(t2, color="black", linestyle="--", label="t*")
    plt.xlabel("t")
    plt.ylabel("variance")
    plt.legend()
    plt.savefig("./plots/2.png")
    plt.close()

    # Question 3
    img2 = skimage.io.imread("./images/sudoku.png")

    plt.title("Binarization on Full Image")
    plt.imshow(adaptive_binarization(img2, 1), "gray")
    plt.savefig("3a.png")
    plt.close()

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("Adaptive Binarization")
    k = 1
    for i in range(2):
        for j in range(2):
            b = 2**k
            axs[i, j].imshow(adaptive_binarization(img2, b), "gray")
            axs[i, j].set_title(f'{b} x {b}')
            k += 1
    fig.tight_layout()
    plt.savefig("./plots/3b.png")
    plt.close()

    # Question 4
    img3 = skimage.io.imread("./images/quote.png")
    img3_binarized = adaptive_binarization(img3, 1)
    k, connected = connected_components(img3_binarized)
    print(f"Number of connected components: {k}")
    plt.clf()
    plt.cla()
    plt.imshow(connected)
    plt.savefig("./plots/4.png")
    plt.close()