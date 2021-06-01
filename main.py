import matplotlib.pyplot as plt
import numpy as np
import torch


def get_data(batch_size):
    data = []
    targets = np.zeros(batch_size)
    for batch_i in range(batch_size):
        x_point = [np.random.uniform(-1.5, 3.5), np.random.uniform(-10., 10.)]
        data.append(x_point)
        if x_point[0] >= 2:
            targets[batch_i] = 0
        elif 1 <= x_point[0] < 2:
            targets[batch_i] = 1
        elif 0 <= x_point[0] < 1:
            targets[batch_i] = 2
        elif x_point[0] < 0:
            targets[batch_i] = 3
    return torch.FloatTensor(data), torch.LongTensor(targets)


def get_x_y_from_tensor(tensor):
    x_points = []
    y_points = []
    for i in range(tensor.size()[0]):
        x_points.append(tensor[i][0].item())
        y_points.append(tensor[i][1].item())
    return x_points, y_points


def accuracy_score(preds, targets):
    preds = preds.data.numpy()
    preds = np.argmax(preds, axis=1)
    total = len(targets.data.numpy())

    correct = 0
    for i in range(len(targets)):
        if preds[i] == targets[i]:
            correct += 1

    # print("Целевые переменные: ", targets.data.numpy())
    # print("Предсказания модели:", preds)
    print("Точность: ", correct / total)


def point_classification(test_data, preds):
    first_class_x_points = []
    first_class_y_points = []

    second_class_x_points = []
    second_class_y_points = []

    third_class_x_points = []
    third_class_y_points = []

    fourth_class_x_points = []
    fourth_class_y_points = []

    preds = preds.data.numpy()
    preds = np.argmax(preds, axis=1)

    for i in range(test_data.size()[0]):
        if preds[i] == 0:
            first_class_x_points.append(test_data[i][0].item())
            first_class_y_points.append(test_data[i][1].item())
        elif preds[i] == 1:
            second_class_x_points.append(test_data[i][0].item())
            second_class_y_points.append(test_data[i][1].item())
        elif preds[i] == 2:
            third_class_x_points.append(test_data[i][0].item())
            third_class_y_points.append(test_data[i][1].item())
        elif preds[i] == 3:
            fourth_class_x_points.append(test_data[i][0].item())
            fourth_class_y_points.append(test_data[i][1].item())
    return first_class_x_points, first_class_y_points, second_class_x_points, second_class_y_points, third_class_x_points, third_class_y_points, fourth_class_x_points, fourth_class_y_points


def visualize_data(train_data, preds):
    x1_points, y1_points, x2_points, y2_points, x3_points, y3_points, x4_points, y4_points = point_classification(
        train_data, preds)

    classes = preds.data.numpy()
    classes = np.argmax(classes, axis=1)

    x, y = get_x_y_from_tensor(train_data)

    plt.figure(1)
    plt.grid()

    plt.xlabel("$x$", fontsize=14)
    plt.ylabel("$y$", fontsize=14)

    plt.plot([2, 2], [-11, 11])
    plt.plot([1, 1], [-11, 11])
    plt.plot([0, 0], [-11, 11])
    #plt.plot([-1.5, -1.5], [-11, 11])

    plt.scatter(x, y)

    plt.figure(2)

    plt.plot([2, 2], [-11, 11])
    plt.plot([1, 1], [-11, 11])
    plt.plot([0, 0], [-11, 11])
    #plt.plot([-1.5, -1.5], [-11, 11])

    plt.scatter(x1_points, y1_points, label="1 класс")
    plt.scatter(x2_points, y2_points, label="2 класс")
    plt.scatter(x3_points, y3_points, label="3 класс")
    plt.scatter(x4_points, y4_points, label="4 класс")
    plt.legend()

    plt.grid()
    plt.xlabel("$x$", fontsize=14)
    plt.ylabel("$y$", fontsize=14)

    plt.show()


batch_size = 70
input_dim = 2
hidden_dimension = 10
output_dimension = 4
learning_rate = 0.01
epochs = 7500

two_layer_net = torch.nn.Sequential(
    torch.nn.Linear(input_dim, hidden_dimension),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dimension, output_dimension)
)

ce_loss = torch.nn.CrossEntropyLoss(size_average=False)
optimizer = torch.optim.SGD(two_layer_net.parameters(), lr=learning_rate)
print("Обучение запущено")
for epoch in range(epochs):
    train_data, train_targets = get_data(batch_size)
    preds = two_layer_net(train_data)

    loss = ce_loss(preds, train_targets)
    optimizer.zero_grad()

    # print(loss.item())

    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(epoch, "эпоха")

print("Обучение окончено")

test_data, test_targets = get_data(200)
check = torch.FloatTensor([[0.99, 10]])
preds = two_layer_net(test_data)

accuracy_score(preds, test_targets)

visualize_data(test_data, preds)
