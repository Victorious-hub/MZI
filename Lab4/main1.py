import random
import numpy as np


class McEliece:
    def __init__(self):
        self.H = np.array(
            [
                [1, 0, 1, 0, 1, 0, 1],
                [0, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 1, 1, 1, 1]
            ]
        )

        self.G = np.array(
            [
                [1, 1, 0, 1],
                [1, 0, 1, 1], 
                [1, 0, 0, 0],
                [0, 1, 1, 1],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        )

        # decode
        self.R = np.array(
            [
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1]
            ]
        )
        self.P = self.generate_permutation_matrix(7)
        self.P_inv = self.P.T
        self.S = self.random_binary_non_singular_matrix(4) # невырожденная матрица
        self.S_inv = self.gf2_matrix_inverse(self.S)


    def gf2_matrix_inverse(self, a: np.ndarray) -> np.ndarray:
        """Compute inverse of binary matrix a over GF(2). Raises ValueError if not invertible."""
        n = a.shape[0]
        A = a.copy() % 2
        inv = np.eye(n, dtype=int)
        for col in range(n):
            # find pivot
            pivot = None
            for row in range(col, n):
                if A[row, col] == 1:
                    pivot = row
                    break
            if pivot is None:
                raise ValueError("Matrix not invertible over GF(2)")
            if pivot != col:
                # swap rows
                A[[col, pivot]] = A[[pivot, col]]
                inv[[col, pivot]] = inv[[pivot, col]]
            # eliminate other rows
            for row in range(n):
                if row != col and A[row, col] == 1:
                    A[row] ^= A[col]
                    inv[row] ^= inv[col]
        return inv % 2

    def random_binary_non_singular_matrix(self, n):
        while True:
            a = np.random.randint(0, 2, size=(n, n)).astype(int)
            try:
                _ = self.gf2_matrix_inverse(a)
                return a
            except ValueError:
                continue

    def generate_permutation_matrix(self, n):
        perm = np.random.permutation(n)
        P = np.zeros((n, n), dtype=int)
        for i, j in enumerate(perm):
            P[i, j] = 1
        return P

    def detect_error(self, err_enc_bits):
        err_idx_vec = np.mod(self.H.dot(err_enc_bits), 2)
        err_idx_vec = err_idx_vec[::-1]
        err_idx = int(''.join(str(bit) for bit in err_idx_vec), 2)
        return err_idx - 1

    def hamming7_4_encode(self, p_str):
        p = np.array([int(x) for x in p_str]) # M
        
        G_hat = np.transpose(np.mod((self.S.dot(np.transpose(self.G))).dot(self.P), 2)) # это и есть публичный ключ
        prod = np.mod(G_hat.dot(p), 2)
        print(G_hat)
        return prod
        # M * G'

    def hamming7_4_decode(self, c):
        prod = np.mod(self.R.dot(c), 2)
        return prod

    def flip_bit(self, bits, n):
        bits[n] = (bits[n] + 1) % 2

    def add_single_bit_error(self, enc_bits):
        error = [0] * 7
        idx = random.randint(0, 6)
        error[idx] = 1
        return np.mod(enc_bits + error, 2) #

    def split_binary_string(self, str, n):
        return [str[i:i + n] for i in range(0, len(str), n)]

    def bits_to_bytes(self, bits):
        if len(bits) % 8 != 0:
            bits += '0' * (8 - len(bits) % 8)
        byte_chunks = [bits[i:i+8] for i in range(0, len(bits), 8)]
        return bytes([int(b, 2) for b in byte_chunks])


if __name__ == '__main__':
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    text_bytes = text.encode('utf-8')
    binary_str = ''.join(format(b, '08b') for b in text_bytes)

    algo = McEliece()

    print('Read ', "input.txt", '...')
    split_bits_list = algo.split_binary_string(binary_str, 4)
    enc_msg = []
    for split_bits in split_bits_list:
        enc_bits = algo.hamming7_4_encode(split_bits)
        err_enc_bits = algo.add_single_bit_error(enc_bits)

        str_enc = ''.join(str(x) for x in err_enc_bits)
        enc_msg.append(str_enc)

    encoded = ''.join(enc_msg)
    with open("encrypt.txt", "w", encoding="utf-8") as f:
        f.write(encoded)
    print('Write in ', "encrypt.txt", '...')
    dec_msg = []
    for enc_bits in enc_msg:
        enc_bits = np.array([int(x) for x in enc_bits])
        c_hat = np.mod(enc_bits.dot(algo.P_inv), 2)

        err_idx = algo.detect_error(c_hat)
        algo.flip_bit(c_hat, err_idx)
        m_hat = algo.hamming7_4_decode(c_hat)
        m_out = np.mod(m_hat.dot(algo.S_inv), 2)

        str_dec = ''.join(str(x) for x in m_out)
        dec_msg.append(str_dec)

    dec_msg_bytes = algo.bits_to_bytes(''.join(dec_msg))
    txt = dec_msg_bytes.decode('utf-8')
    print(txt)
 
    print('Write in', "decoded.txt", '...')
    with open("decoded.txt", "w", encoding="utf-8") as f:
        f.write(txt)
