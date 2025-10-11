import random

"""
Алгоритм Рабина
1. Генерируем открытый и закрытый ключи p, q с длиной битов(по умолчанию 42)
- числа большие(поэтому биты 42)
- числа простые
- n = p * q - открытый ключ, (p, q) - закрытый ключ
- num % 4 == 3 - важно при вычислении квадратных корней(там формула простоая (p + 1) /4 ) при расшифровании
- Оно также доказывает что будет 4 корня при выборе
"""

class RabinCipher:
    def __init__(self, bits=42):
        self.open_key, self.close_key = self.generate_key(bits)

        # гарантируем, что p != q
        while self.close_key[0] == self.close_key[1]:
            self.open_key, self.close_key = self.generate_key(bits)

    # Реализация расширенного алгоритма Евклида
    # ищем x,y такие что a * x + b * y = gcd(a, b). Нужно для комбинации корней и КТО
    def extended_gcd(self, a, b):
        if a == 0:
            return 0, 1
        else:
            x, y = self.extended_gcd(b % a, a)
            return y - (b // a) * x, x

    # Бинарное возведение в степень
    @staticmethod
    def mod(k, b, m):
        
        i = 0
        a = 1
        v = []
        while k > 0:
            v.append(k % 2)
            k = (k - v[i]) // 2
            i += 1
        for j in range(i):
            if v[j] == 1:
                a = (a * b) % m
                b = (b * b) % m
            else:
                b = (b * b) % m
        return a

    # простой способ через деление sqrt(n). Если числа большие биты, уже ненадежно
    @staticmethod
    def is_prime(n):
        if n <= 1:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

    @classmethod
    def generate_prime(cls, bits):
        while True:
            num = random.getrandbits(bits)
            if num % 4 == 3 and cls.is_prime(num): # числа вида 4k + 3. Нужно для генерации простоты числа
                return num

    @classmethod
    def generate_key(cls, bits):
        p = cls.generate_prime(bits)
        q = cls.generate_prime(bits)
        open_key = p * q
        close_key = (p, q) # tuple for closed key
        return open_key, close_key

    # через gcd находим такие Yp, Yq что Yp * p + Yq * q = 1 - нужно для комбинации корней в декодировании(КТО)
    def find_Yp_Yq(self, p, q):
        x, y = self.extended_gcd(p, q)
        if x < 0:
            x += q
        Yp = x
        Yq = (1 - Yp * p) // q
        return Yp, Yq

    # Для каждого символа текста берёт его код ord(char) → шифрует как c = m**2 mod n
    def encrypt_char(self, char):
        number = ord(char)
        return (number ** 2) % self.open_key

    def decrypt_char(self, c):
        p, q = self.close_key
        x, y = self.find_Yp_Yq(p, q)

        while x * p + y * q != 1:
            x, y = self.find_Yp_Yq(p, q) # решаем расширенный алгоритм Евлкида(коэффициенты находим) 1 = gcd(p,q) == 1 всегда(простые числа)

        # Тут мы находим квадратные корни числа 𝑐 c по модулю 𝑝 и q. x^2 = c (mod p)
        r = self.mod((p + 1) // 4, c, p)
        s = self.mod((q + 1) // 4, c, q)

        r1 = (x * p * s + y * q * r) % self.open_key
        r2 = self.open_key - r1
        r3 = (x * p * s - y * q * r) % self.open_key
        r4 = self.open_key - r3

        # Перебирает эти четыре корня и возвращает первый chr(item), если 0 <= item < 0x110000 (т.е. «возможно валидный Unicode-символ»), иначе возвращает �
        # В реальности здесь нужно юзать падинг, чтобы по смещению искать
        for item in (r1, r2, r3, r4):
            if 0 <= item < 0x110000:  # 0x10FFFF + 1, максимальный код символа в Python
                return chr(item)
        return "�"

with open("input.txt", "r", encoding='utf-8') as f:
    text = f.read()

cipher = RabinCipher(bits=42)

with open("encrypted.txt", "w", encoding='utf-8') as f_enc, \
     open("decrypted.txt", "w", encoding='utf-8') as f_dec:

    for char in text:
        encrypted_char = cipher.encrypt_char(char)
        f_enc.write(str(encrypted_char))
        decrypted_char = cipher.decrypt_char(encrypted_char)
        f_dec.write(decrypted_char)
