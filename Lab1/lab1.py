
# Exchange block (S-block)
blocks = (
    (4, 10, 9, 2, 13, 8, 0, 14, 6, 11, 1, 12, 7, 15, 5, 3),
    (14, 11, 4, 12, 6, 13, 15, 10, 2, 3, 8, 1, 0, 7, 5, 9),
    (5, 8, 1, 13, 10, 3, 4, 2, 14, 15, 12, 7, 6, 0, 9, 11),
    (7, 13, 10, 1, 0, 8, 9, 15, 14, 4, 6, 12, 11, 2, 5, 3),
    (6, 12, 7, 1, 5, 15, 13, 8, 4, 10, 9, 14, 0, 3, 11, 2),
    (4, 11, 10, 0, 7, 2, 1, 13, 3, 6, 8, 5, 9, 12, 15, 14),
    (13, 11, 4, 1, 3, 15, 5, 9, 0, 10, 14, 7, 6, 8, 2, 12),
    (1, 15, 13, 0, 5, 7, 10, 4, 9, 2, 3, 14, 6, 11, 8, 12),
)

key = 101296096738265935819101407525257139131603781785725466795748840673385905361244 
IV = 10916177444472819315

def bit_length(value):
    return len(bin(value)[2:])

class GOST:
    def __init__(self, key, sbox):
        self.sbox = sbox
        self.subkeys = [(key >> (32 * i)) & 0xFFFFFFFF for i in range(8)]

    # Feistel network
    def _f(self, part, key):
        temp = (part + key) & 0xFFFFFFFF # (2**32 - 1)
        out = 0
        for i in range(8):
            out |= (self.sbox[i][(temp >> (4 * i)) & 0xF]) << (4 * i)
        return ((out << 11) | (out >> (32 - 11))) & 0xFFFFFFFF

    def encrypt_block(self, block64):
        n1 = block64 >> 32
        n2 = block64 & 0xFFFFFFFF
        for i in range(24):
            n1, n2 = n2, n1 ^ self._f(n2, self.subkeys[i % 8])
        for i in range(8):
            n1, n2 = n2, n1 ^ self._f(n2, self.subkeys[7 - i])
        return (n1 << 32) | n2

class GammaMode:
    def __init__(self, gost: GOST, iv: int):
        self.gost = gost
        self.counter = iv # iv -  синхропосылка

    def process(self, data: bytes) -> bytes:
        out = bytearray()
        for i in range(0, len(data), 8):
            block = int.from_bytes(data[i:i+8].ljust(8, b'\0'), "big")
            gamma = self.gost.encrypt_block(self.counter)
            self.counter = (self.counter + 1) & 0xFFFFFFFFFFFFFFFF
            xored = block ^ gamma
            out.extend(xored.to_bytes(8, "big")[:len(data[i:i+8])])
        return bytes(out)

def main():
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    gost = GOST(key, blocks)
    gamma = GammaMode(gost, IV)

    print(text)
    encrypted = gamma.process(text.encode("utf-8"))

    with open("encr.txt", "w") as file:
        file.write(encrypted.hex())
    print("Файл зашифрован")

    with open("encr.txt", "r") as file:
        encrypted_hex = file.read().strip()
    encrypted_bytes = bytes.fromhex(encrypted_hex)
    gamma2 = GammaMode(gost, IV)
    print(encrypted_bytes)
    decrypted = gamma2.process(encrypted_bytes)
    decrypted_str = decrypted.decode("utf-8")
    print(decrypted_str)

    with open("decr.txt", "w", encoding="utf-8") as file:
        file.write(decrypted_str)
    print("Файл расшифрован")

if __name__ == "__main__":
    main()
