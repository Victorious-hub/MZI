import random
import numpy as np


class McEliece:
    def __init__(self):
        # H — проверочная матрица Хэмминга (7,4)-кода, исправляющая 1 ошибку.
        self.H = np.array(
            [
                [1, 0, 1, 0, 1, 0, 1],
                [0, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 1, 1, 1, 1]
            ]
        )

        # G — порождающая матрица кода (7 бит на выходе из 4 бит на входе).
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

        """Эта матрица используется для декодирования закодированных сообщений.
        Она позволяет извлечь исходные 4 бита из 7-битного закодированного сообщения."""
        # (часть секрета)
        self.R = np.array(
            [
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1]
            ]
        )
        # (часть секрета)
        self.P = self.generate_permutation_matrix(7) # матрица перестановки(также невырожденная)
        self.P_inv = self.P.T  # inverse of a permutation matrix is its transpose
        # (часть секрета)
        self.S = self.random_binary_non_singular_matrix(4) # невырожденная матрица
        self.S_inv = self.gf2_matrix_inverse(self.S)

        # (S, P, G ) - secret key

        # inv нужен для декодирования

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
        # create random invertible binary matrix over GF(2) using Gauss elimination
        while True:
            a = np.random.randint(0, 2, size=(n, n)).astype(int)
            try:
                _ = self.gf2_matrix_inverse(a)
                return a
            except ValueError:
                continue

    def generate_permutation_matrix(self, n):
        # generate permutation matrix (rows permuted identity)
        perm = np.random.permutation(n)
        P = np.zeros((n, n), dtype=int)
        for i, j in enumerate(perm):
            P[i, j] = 1
        return P

    # Определяет позицию ошибки в закодированных данных. Через проверочную матрицу ищем закодированный бит и юзаем flip
    def detect_error(self, err_enc_bits):
        err_idx_vec = np.mod(self.H.dot(err_enc_bits), 2)
        err_idx_vec = err_idx_vec[::-1]
        err_idx = int(''.join(str(bit) for bit in err_idx_vec), 2)
        return err_idx - 1

    # generate public key
    def hamming7_4_encode(self, p_str):
        p = np.array([int(x) for x in p_str]) # M
        
        #Эта матрица выглядит как "произвольный линейный код", и по ней невозможно эффективно декодировать без знания 𝑆, 𝑃.
        G_hat = np.transpose(np.mod((self.S.dot(np.transpose(self.G))).dot(self.P), 2)) # это и есть публичный ключ
        prod = np.mod(G_hat.dot(p), 2) # это как раз шифрование блока сообщения.
        print(G_hat)
        return prod # кодовый вектор 7 битов
        # M * G'

    # Используем матрицу R, чтобы извлечь исходные 4 информационных бита, но они ещё «замаскированы» матрицей S.
    def hamming7_4_decode(self, c):
        prod = np.mod(self.R.dot(c), 2)
        return prod

    # flip_bit меняет этот бит → теперь у нас исправленное кодовое слово:
    def flip_bit(self, bits, n):
        bits[n] = (bits[n] + 1) % 2

    # здесь мы вставляем блок с ошибкой(с 1 битом ошибки) - это нужно для криптостойкости чтобы расшифровать было сложнее линейный код
    # # M * G' + Z (вектор ошибки)
    def add_single_bit_error(self, enc_bits):
        error = [0] * 7
        idx = random.randint(0, 6)
        error[idx] = 1
        return np.mod(enc_bits + error, 2) #

    def split_binary_string(self, str, n):
        return [str[i:i + n] for i in range(0, len(str), n)]

    def bits_to_bytes(self, bits):
        # Дополняем строку до кратной 8 бит
        if len(bits) % 8 != 0:
            bits += '0' * (8 - len(bits) % 8)
        byte_chunks = [bits[i:i+8] for i in range(0, len(bits), 8)]
        return bytes([int(b, 2) for b in byte_chunks])


if __name__ == '__main__':
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    text_bytes = text.encode('utf-8')  # UTF-8 кодировка для кириллицы
    binary_str = ''.join(format(b, '08b') for b in text_bytes)

    algo = McEliece()

    # k = 4 по коду Хемминга разбиваем на блоки по 4
    # длина кодового слова (выхода после кодирования). = 7 по коду Хемминга
    print('Read ', "input.txt", '...')
    # разделяем биты по 4 чанка
    split_bits_list = algo.split_binary_string(binary_str, 4)
    enc_msg = []
    for split_bits in split_bits_list:
        enc_bits = algo.hamming7_4_encode(split_bits) # Кодирование
        # добавляем рандомную ошибку
        err_enc_bits = algo.add_single_bit_error(enc_bits)

        # конвертируем в строку и добавляем к результату
        str_enc = ''.join(str(x) for x in err_enc_bits)
        enc_msg.append(str_enc)

    encoded = ''.join(enc_msg)
    with open("encrypt.txt", "w", encoding="utf-8") as f:
        f.write(encoded)
    print('Write in ', "encrypt.txt", '...')
    dec_msg = []
    for enc_bits in enc_msg:
        enc_bits = np.array([int(x) for x in enc_bits])
        # Вычисляем c_hat = c * P_inv, здесь мы разворавчиваем перестановку, нам надо вернуться в исходное положение с M * S * G) + Z
        c_hat = np.mod(enc_bits.dot(algo.P_inv), 2) # теперь кодовое слово «правильной структуры», можно искать синдром и исправлять ошибки.

        # находим бит ошибки
        err_idx = algo.detect_error(c_hat)
        # переворачиваем бит ошибки
        algo.flip_bit(c_hat, err_idx)
        # находим m_hat
        m_hat = algo.hamming7_4_decode(c_hat)
        # находим m = m_hat * S_inv
        # Умножаем на обратную матрицу 𝑆*-1: Теперь у нас восстановлен оригинальный 4-битный блок сообщения.
        m_out = np.mod(m_hat.dot(algo.S_inv), 2)

        # Все 4-битные блоки соединяются в одну бинарную строку.
        # Затем переводим обратно в символы (ASCII) через функцию bits_to_str.
        #  В результате получаем исходный текст, полностью совпадающий с тем, что было на входе.
        str_dec = ''.join(str(x) for x in m_out)
        dec_msg.append(str_dec)

    # dec_msg is a list of 4-bit strings; join into one binary string before converting to bytes
    dec_msg_bytes = algo.bits_to_bytes(''.join(dec_msg))
    txt = dec_msg_bytes.decode('utf-8')
    print(txt)
 
    print('Write in', "decoded.txt", '...')
    with open("decoded.txt", "w", encoding="utf-8") as f:
        f.write(txt)
