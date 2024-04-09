import sys
import librosa
import matplotlib.pyplot as plt


def find_instr(wav_file: str):

    data, sample_rate = librosa.load(wav_file)

    # t = range(len(data))
    # plt.plot(t, data)
    # plt.show()

    spectral_centroid = librosa.feature.spectral_centroid(
        y=data, sr=sample_rate)[0]

    mean = sum(spectral_centroid) / len(spectral_centroid)

    # t = range(len(spectral_centroid))
    # plt.plot(t, spectral_centroid)
    # plt.show()

    # print("SR = ", mean)

    if mean < 2000:
        return "guitar"
    elif mean > 3500:
        return "drums"
    else:
        return "both"


def test_list(l):
    for path in l:
        wav_path = f"./instr/{path}.wav"
        INSTR = find_instr(wav_path)
        print(f"{path}\t=> {INSTR}")


if len(sys.argv) < 2:
    G = [f"g{i}" for i in range(1, 18)]
    D = [f"d{i}" for i in range(1, 14)]
    DBB = [f"dbb{i}" for i in range(1, 5)] + [f"s{i}" for i in range(1, 5)]
    print("\n\n-----GUITAR-----")
    test_list(G)
    print("\n\n-----DRUMS-----")
    test_list(D)
    print("\n\n-----BOTH-----")
    test_list(DBB)


else:
    INSTR = find_instr(f"./instr/{sys.argv[1]}.wav")
    print(f"INSTR = {INSTR}")
