import matplotlib.pyplot as plt
import numpy as np

# 기존 코드 유지
fft_1 = np.zeros((32, 32, 32))

# 설정값
height, width = 32, 32
x_min = np.random.randint(width // 32, width // 2)
x_max = np.random.randint(width // 2, width - width // 32)
y_min = np.random.randint(height // 32, height // 2)
y_max = np.random.randint(height // 2, height - height // 32)

x_min_2 = np.random.randint(width // 32, x_min + 1)
x_max_2 = np.random.randint(x_max, width - width // 32)
y_min_2 = np.random.randint(height // 32, y_min + 1)
y_max_2 = np.random.randint(y_max, height - height // 32)

matrix_1 = fft_1[x_min:x_max, y_min:y_max]
matrix_2 = fft_1[x_min_2:x_max_2, y_min_2:y_max_2]

B = 0.5
b = np.random.uniform(0, B)
array3 = np.random.uniform(1 - b, 1 + b, size=fft_1.shape)

A = 5
a = np.random.uniform(0, A)
array1 = np.random.uniform(-a, a, size=matrix_1.shape)

fft_1 = fft_1 * array3
fft_1[x_min_2:x_max_2, y_min_2:y_max_2] = matrix_2
fft_1[x_min:x_max, y_min:y_max] = matrix_1 * array1

# 평균값 계산
mean_projection = fft_1.mean(axis=2)  # Z축 평균값

# 선 그래프 시각화
plt.figure(figsize=(10, 6))

# 행별 평균값 흐름
plt.plot(np.arange(mean_projection.shape[0]), mean_projection.mean(axis=1), marker='o', label="Row-wise Mean")

# 열별 평균값 흐름
plt.plot(np.arange(mean_projection.shape[1]), mean_projection.mean(axis=0), marker='x', label="Column-wise Mean")

# 그래프 설정
plt.title("Value Flow in Mean Projection")
plt.xlabel("Index")
plt.ylabel("Mean Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 확인용 출력
print(f"x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")
