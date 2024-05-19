import tkinter as tk

import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


def butter_bandpass_filter(signal, low, high, rate, order):
    # bandpass filter using a Butterworth filter to remove noise outside a specific frequency range (0.5 Hz to 20 Hz)
    # rate :  sampling rate
    nyq = 0.5 * rate  # calculates the Nyquist frequency (nyq) as half of the sampling rate.
    l = low / nyq  # lower cutoff frequency (normalize)
    h = high / nyq  # high cutoff frequency(normalize)
    n, d = butter(order, [l, h], btype='band', output='ba', analog=False, fs=None)  # design the Butterworth filter coefficients (n and d)
    filterd = filtfilt(n, d, signal)  # applies the filter to the input signal using the filtfilt function
    return filterd


def preprocessing(data):
    filterd_x = []  # will be used to store the filtered signals

    # apply filter on each signal
    #data.shape[1]: iterates over the columns of the data input
    for i in range(data.shape[1]):
        # selects the first 250 samples
        signal = data[i].iloc[:250]
        # apply a Butterworth bandpass filter to the selected signal segment
        filterd_x.append(butter_bandpass_filter(signal, 0.5, 20, 176, 4))
    return filterd_x


def feature_extract_statistical(data):
    statistical_features = []  # will be used to store the computed statistical features.

    for signal in data.values:
        mean = np.mean(signal)
        std = np.std(signal)
        variance = np.var(signal)  # calculates the variance of the signal.
        energy = np.sum(np.square(signal))  # energy of the signal, which is the sum of squared values.
        statistical_features.append([mean, std, variance, energy])
    return statistical_features


def feature_extract_wavelet(data):
    coeffs_by_signal = []  # will be used to store the wavelet coefficients of each signal.
    wavelet = 'db4'
    levels = 2  # number of decomposition levels for the DWT
    for signal in data.values:
        coeffs = pywt.wavedec(signal, wavelet, level=levels)
        coeffs_by_signal.append(coeffs[0])
    return coeffs_by_signal


train_h = pd.read_excel('train_h.xlsx', header=None)
train_v = pd.read_excel('train_v.xlsx', header=None)
test_h = pd.read_excel('test_h.xlsx', header=None)
test_v = pd.read_excel('test_v.xlsx', header=None)

# Load the labels
train_labels = pd.read_excel('train_h.xlsx', header=None).iloc[-1, :]
test_labels = pd.read_excel('test_h.xlsx', header=None).iloc[-1, :]

train_h_pre = pd.DataFrame(preprocessing(train_h))
train_v_pre = pd.DataFrame(preprocessing(train_v))
test_h_pre = pd.DataFrame(preprocessing(test_h))
test_v_pre = pd.DataFrame(preprocessing(test_v))

train_h_stat = pd.DataFrame(feature_extract_statistical(train_h_pre))
train_v_stat = pd.DataFrame(feature_extract_statistical(train_v_pre))
test_h_stat = pd.DataFrame(feature_extract_statistical(test_h_pre))
test_v_stat = pd.DataFrame(feature_extract_statistical(test_v_pre))

train_h_w = pd.DataFrame(feature_extract_wavelet(train_h_pre))
train_v_w = pd.DataFrame(feature_extract_wavelet(train_v_pre))
test_h_w = pd.DataFrame(feature_extract_wavelet(test_h_pre))
test_v_w = pd.DataFrame(feature_extract_wavelet(test_v_pre))

train_h_stat.to_excel('train_h_statistical_features.xlsx', header=['mean_h', 'std_h', 'variance_h', 'energy_h'],
                      index=False)
train_v_stat.to_excel('train_v_statistical_features.xlsx', header=['mean_v', 'std_v', 'variance_v', 'energy_v'],
                      index=False)
test_h_stat.to_excel('test_h_statistical_features.xlsx', header=['mean_h', 'std_h', 'variance_h', 'energy_h'],
                     index=False)
test_v_stat.to_excel('test_v_statistical_features.xlsx', header=['mean_v', 'std_v', 'variance_v', 'energy_v'],
                     index=False)

train_h_w.to_excel('train_h_wavelet.xlsx', header=False, index=False)
train_v_w.to_excel('train_v_wavelet.xlsx', header=False, index=False)
test_h_w.to_excel('test_h_wavelet.xlsx', header=False, index=False)
test_v_w.to_excel('test_v_wavelet.xlsx', header=False, index=False)

train_stat = pd.concat([train_h_stat, train_v_stat], ignore_index=True, axis=1)
test_stat = pd.concat([test_h_stat, test_v_stat, test_labels], ignore_index=True, axis=1)
train_wavelet = pd.concat([train_h_w, train_v_w], ignore_index=True, axis=1)
test_wavelet = pd.concat([test_h_w, test_v_w, test_labels], ignore_index=True, axis=1)

test_stat = test_stat.sample(frac=1, random_state=42)
test_wavelet = test_wavelet.sample(frac=1, random_state=42)

train_stat.to_excel('train_statistical_features.xlsx',
                    header=['mean_h', 'std_h', 'variance_h', 'energy_h', 'mean_v', 'std_v', 'variance_v', 'energy_v'],
                    index=False)
test_stat.to_excel('test_statistical_features.xlsx',
                   header=['mean_h', 'std_h', 'variance_h', 'energy_h', 'mean_v', 'std_v', 'variance_v', 'energy_v',
                           'class'], index=False)
train_wavelet.to_excel('train_wavelet.xlsx', header=False, index=False)
test_wavelet.to_excel('test_wavelet.xlsx', header=False, index=False)

#-1, selects all columns except the last one
X_test_stat = test_stat.iloc[:, :-1]
#last column of each DataFrame
y_test_stat = test_stat.iloc[:, -1]

X_test_wavelet = test_wavelet.iloc[:, :-1]
y_test_wavelet = test_wavelet.iloc[:, -1]

# Initialize and train a Naive Bayes classifier using statistical features
nb_stat = GaussianNB()
nb_stat.fit(train_stat, train_labels)
y_pred_stat_nb = nb_stat.predict(X_test_stat)
accuracy_stat_nb = accuracy_score(y_test_stat, y_pred_stat_nb)
print("Naive Bayes Accuracy (Statistical Features):", accuracy_stat_nb)

# Initialize and train a Naive Bayes classifier using wavelet features
nb_wavelet = GaussianNB()
nb_wavelet.fit(train_wavelet, train_labels)
y_pred_wavelet_nb = nb_wavelet.predict(X_test_wavelet)
accuracy_wavelet_nb = accuracy_score(y_test_wavelet, y_pred_wavelet_nb)
print("Naive Bayes Accuracy (Wavelet Features):", accuracy_wavelet_nb)

# Initialize and train a random forest classifier using statistical features
random_forest_stat = RandomForestClassifier()
random_forest_stat.fit(train_stat, train_labels)
y_pred_stat_rf = random_forest_stat.predict(X_test_stat)
accuracy_stat_rf = accuracy_score(y_test_stat, y_pred_stat_rf)
print("Random Forest Accuracy (Statistical Features):", accuracy_stat_rf)

# Initialize and train a random forest classifier using wavelet features
random_forest_wavelet = RandomForestClassifier()
random_forest_wavelet.fit(train_wavelet, train_labels)
y_pred_wavelet_rf = random_forest_wavelet.predict(X_test_wavelet)
accuracy_wavelet_rf = accuracy_score(y_test_wavelet, y_pred_wavelet_rf)
print("Random Forest Accuracy (Wavelet Features):", accuracy_wavelet_rf)

y_pred_wavelet = random_forest_wavelet.predict(train_wavelet)
accuracy_wavelet = accuracy_score(train_labels, y_pred_wavelet)
print("Random Forest Accuracy (Wavelet Features) train :", accuracy_wavelet)


ui_test = pd.read_excel('ui_test.xlsx', header=None)
ui_test = ui_test.iloc[:, :-1]
#random_forest_wavelet to make predictions on the ui_test DataFrame.
y_rf = random_forest_wavelet.predict(ui_test)
print(y_rf)


def set_current_button(button):
    global current_button
    current_button = button


def move_up():
    up_button.config(bg="yellow")
    root.after(200, lambda: up_button.config(bg="SystemButtonFace"))


def move_down():
    down_button.config(bg="yellow")
    root.after(200, lambda: down_button.config(bg="SystemButtonFace"))


def move_left():
    left_button.config(bg="yellow")
    root.after(200, lambda: left_button.config(bg="SystemButtonFace"))


def move_right():
    right_button.config(bg="yellow")
    root.after(200, lambda: right_button.config(bg="SystemButtonFace"))


def reset_button_color(button):
    button.config(bg="SystemButtonFace")


def exit_app():
    root.destroy()


def process_array(index):
    if index >= len(y_rf):
        return
    if y_rf[index] == 0:

        if (current_button == down_button):
            print("exit")
            down_button.config(bg="SystemButtonFace")
            set_current_button(exit_button)
            exit_button.config(bg="green")
        else:
            print("down")
            set_current_button(down_button)
            down_button.config(bg="green")
    elif y_rf[index] == 1:
        print("blink")
        current_button.invoke()
    elif y_rf[index] == 2:
        print("right")
        set_current_button(right_button)
        right_button.config(bg="green")
    elif y_rf[index] == 3:
        print("left")
        set_current_button(left_button)
        left_button.config(bg="green")
    elif y_rf[index] == 4:
        print("up")
        set_current_button(up_button)
        up_button.config(bg="green")
    root.after(1000, process_array, index + 1)


root = tk.Tk()
root.title("Arrow Buttons")

# Create buttons for arrow directions with centered arrows
up_button = tk.Button(root, text="\u2191", command=move_up, font=("Arial", 16))
down_button = tk.Button(root, text="\u2193", command=move_down, font=("Arial", 16))
left_button = tk.Button(root, text="\u2190", command=move_left, font=("Arial", 16))
right_button = tk.Button(root, text="\u2192", command=move_right, font=("Arial", 16))
exit_button = tk.Button(root, text="Exit", command=exit_app)

# Arrange the buttons using grid layout
up_button.grid(row=0, column=1, padx=10, pady=10)
down_button.grid(row=2, column=1, padx=10, pady=10)
left_button.grid(row=1, column=0, padx=10, pady=10)
right_button.grid(row=1, column=2, padx=10, pady=10)
exit_button.grid(row=4, column=1, padx=10, pady=10)

# Simulating the y_rf array

process_array(0)
root.mainloop()
