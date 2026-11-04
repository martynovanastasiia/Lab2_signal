import numpy as np
import matplotlib.pyplot as plt
import time

def calc_one_term(f_i, k, i, N):
    angle = -2 * np.pi * k * i / N
    return (f_i / N) * np.exp(1j * angle)

def calc_C_k(f, k, N):
    C_k = 0j
    for i in range(N):
        C_k += calc_one_term(f[i], k, i, N)
    return C_k

def part_one(n_variant=18):
    N = 10 + n_variant
    print(f"Кількість значень сигналу N = {N}")
    
    f = np.random.rand(N) * 10 
    
    start_time = time.perf_counter()
    
    C = np.zeros(N, dtype=complex)
    additions = 0
    multiplications = 0
    
    for k in range(N):
        C[k] = calc_C_k(f, k, N)
        multiplications += 2 * N
        additions += N
        
    execution_time = time.perf_counter() - start_time
    
    amplitudes = np.abs(C)
    phases = np.angle(C)
    
    print(f"Час обчислення ДПФ: {execution_time:.6f} секунд")
    print(f"Орієнтовна кількість операцій множення: {multiplications}")
    print(f"Орієнтовна кількість операцій додавання: {additions}\n")

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.stem(range(N), amplitudes)
    plt.title("Спектр амплітуд $|C_k|$")
    plt.xlabel("k (номер гармоніки)")
    plt.ylabel("Амплітуда")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.stem(range(N), phases)
    plt.title("Спектр фаз $arg(C_k)$")
    plt.xlabel("k (номер гармоніки)")
    plt.ylabel("Фаза (радіани)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()



def part_two_and_three(n_variant=18):
    N_number = 96 + n_variant
    print(f"Число N для Частини II = {N_number}")
    
    s_discrete = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    N = len(s_discrete)
    
    C = np.zeros(N, dtype=complex)
    for k in range(N):
        C[k] = calc_C_k(s_discrete, k, N)
        
    for k in range(N//2 + 1):
        mag = np.abs(C[k])
        phase = np.angle(C[k])
        print(f"C_{k}: значення = {C[k]:.4f}, модуль = {mag:.4f}, аргумент = {phase:.4f} рад")
        
    print("\nАналітичний вираз первинного аналогового сигналу s(t):")
    s_t_str = f"s(t) = {np.abs(C[0]):.4f}"
    for k in range(1, N//2):
        s_t_str += f" + {2*np.abs(C[k]):.4f}*cos({2*k}*pi*t/Tc + ({np.angle(C[k]):.4f}))"
    s_t_str += f" + {np.abs(C[N//2]):.4f}*cos({2*(N//2)}*pi*t/Tc + ({np.angle(C[N//2]):.4f}))"
    print(s_t_str)
    
    t_continuous = np.linspace(0, 8, 500)
    s_analog = np.ones_like(t_continuous) * np.abs(C[0])
    
    for k in range(1, N//2):
        s_analog += 2 * np.abs(C[k]) * np.cos(2 * np.pi * k * t_continuous + np.angle(C[k]))
    s_analog += np.abs(C[N//2]) * np.cos(2 * np.pi * (N//2) * t_continuous + np.angle(C[N//2]))

    plt.figure(figsize=(10, 5))
    plt.plot(t_continuous, s_analog, label="Відтворений аналоговий сигнал $s(t)$", color='blue')
    t_discrete = np.arange(N) / N
    plt.stem(t_discrete, s_discrete, linefmt='r--', markerfmt='ro', label="Дискретні відліки")
    plt.title("Відтворення аналогового сигналу на підставі ДПФ")
    plt.xlabel("$t / T_c$")
    plt.ylabel("$s(t)$")
    plt.grid(True)
    plt.legend()
    plt.show()



    print("\nПеревірка ОДПФ")
    print("Аналітичні вирази відліків при n = 0 та n = 1:")
    
    print("\nДля n = 0:")
    print("s(0) = C_0 + 2*Re(C_1) + 2*Re(C_2) + 2*Re(C_3) + C_4")
    term0_1 = 2 * C[1].real
    term0_2 = 2 * C[2].real
    term0_3 = 2 * C[3].real
    term0_4 = C[4].real
    print(f"s(0) = {C[0].real:.4f} + ({term0_1:.4f}) + ({term0_2:.4f}) + ({term0_3:.4f}) + ({term0_4:.4f})")
    s_0_reconstructed = C[0].real + term0_1 + term0_2 + term0_3 + term0_4
    print(f"Обчислене s(0) = {s_0_reconstructed:.4f} (Оригінал: {s_discrete[0]})")
    
    print("\nДля n = 1:")
    print("s(1) = C_0 + 2|C_1|cos(2*pi*1/8 + arg(C_1)) + 2|C_2|cos(4*pi*1/8 + arg(C_2)) + 2|C_3|cos(6*pi*1/8 + arg(C_3)) + C_4*cos(pi)")
    term1_1 = 2 * np.abs(C[1]) * np.cos(2*np.pi*1/8 + np.angle(C[1]))
    term1_2 = 2 * np.abs(C[2]) * np.cos(4*np.pi*1/8 + np.angle(C[2]))
    term1_3 = 2 * np.abs(C[3]) * np.cos(6*np.pi*1/8 + np.angle(C[3]))
    term1_4 = C[4].real * np.cos(np.pi)
    print(f"s(1) = {C[0].real:.4f} + ({term1_1:.4f}) + ({term1_2:.4f}) + ({term1_3:.4f}) + ({term1_4:.4f})")
    s_1_reconstructed = C[0].real + term1_1 + term1_2 + term1_3 + term1_4
    print(f"Обчислене s(1) = {s_1_reconstructed:.4f} (Оригінал: {s_discrete[1]})")

if __name__ == "__main__":
    part_one(n_variant=18)
    part_two_and_three(n_variant=18)