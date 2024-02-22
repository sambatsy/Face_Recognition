class CNN:
    def __init__(self):
        self.model = Sequential()

    def build_model(self):
        self.model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(64, 64, 1), padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='sigmoid'))
        self.model.add(Dense(40, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def summary(self):
        self.model.summary()

    def train(self, X_train, Y_train, X_val, Y_val, batch_size, epochs):
        H = self.model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=batch_size, epochs=epochs, verbose=1)
        return H

cnn_model = CNN()
cnn_model.build_model()
#cnn_model.summary()

H2 = cnn_model.train(X_train, Y_train, X_val, Y_val, batch_size=16, epochs=10)
