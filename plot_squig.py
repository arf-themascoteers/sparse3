import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data/indian_pines_min.csv")
df = df.iloc[:, :-1]
scaler = MinMaxScaler()
df = scaler.fit_transform(df)
signal = df[51]

# cmap = plt.get_cmap('viridis')
# colors = cmap(signal)
#
# fig, ax = plt.subplots()
#
# for i in range(len(signal) - 1):
#     ax.plot([i, i + 1], [0.5, 0.5], color=colors[i], linewidth =10)
#
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
# sm.set_array([])
# fig.colorbar(sm, ax=ax, label='Magnitude')
#
# plt.show()

df = pd.read_csv("results/10_zhang_indian_pines_5_0.csv")
column_index = df.columns.get_loc("weight_0")
df = df.iloc[:, column_index:column_index+200]
weights = df.iloc[-1].to_numpy()
mod_signal = signal * weights

cmap = plt.get_cmap('viridis')
colors = cmap(mod_signal)

fig, ax = plt.subplots()

for i in range(len(signal) - 1):
    #ax.plot([i, i + 1], [mod_signal[i], mod_signal[i + 1]], color=colors[i])
    ax.plot([i, i + 1], [0.5, 0.5], color=colors[i], linewidth =10)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Magnitude')

plt.show()

