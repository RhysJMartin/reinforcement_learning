import pickle

batch = pickle.load(open('batch.p', 'rb'))

print(batch)

position = []
speed = []
for i in range(len(batch)):
    position.append(batch[i][0][0])
    speed.append((batch[i][0][1]))

print('position {}, {}, speed: {}, {}'.format(min(position), max(position), min(speed), max(speed)))