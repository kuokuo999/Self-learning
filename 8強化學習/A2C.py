import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt  # 引入matplotlib繪製圖形

# 創建 CartPole-v1 環境，並指定渲染模式
env = gym.make('CartPole-v1', render_mode='human')


# 定義模型架構
class ActorCritic(tf.keras.Model):
    def __init__(self, action_space):
        super(ActorCritic, self).__init__()
        self.action_space = action_space

        # Actor 網絡：預測行為的機率分佈
        self.actor = tf.keras.Sequential([
            layers.InputLayer(input_shape=(env.observation_space.shape[0],)),
            layers.Dense(128, activation='relu'),
            layers.Dense(action_space, activation='softmax')
        ])

        # Critic 網絡：預測價值函數
        self.critic = tf.keras.Sequential([
            layers.InputLayer(input_shape=(env.observation_space.shape[0],)),
            layers.Dense(128, activation='relu'),
            layers.Dense(1)
        ])

    def call(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value


# 定義 A2C 的訓練步驟
def train_step(model, state, action, reward, next_state, done, gamma=0.99):
    with tf.GradientTape() as tape:
        # 計算行為機率和價值
        action_probs, value = model(state)

        # 計算行為的機率分佈和價值
        action_prob = tf.reduce_sum(action_probs * action, axis=1, keepdims=True)
        next_value = model.critic(next_state)

        # 計算 TD error
        target = reward + gamma * next_value * (1 - done)
        delta = target - value

        # 計算 Actor 和 Critic 的損失
        actor_loss = -tf.math.log(action_prob) * delta
        critic_loss = delta ** 2

        # 總損失
        total_loss = tf.reduce_mean(actor_loss + critic_loss)

    # 計算梯度並更新權重
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss


# 設置參數
action_space = env.action_space.n
state_dim = env.observation_space.shape[0]
model = ActorCritic(action_space)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 訓練模型
num_episodes = 1000
episode_rewards = []  # 用來存儲每個回合的總回報

for episode in range(num_episodes):
    state, _ = env.reset()  # 獲取狀態和額外信息
    state = np.expand_dims(state, axis=0)
    done = False
    episode_reward = 0

    while not done:
        # 生成行為
        action_probs, _ = model(state)
        action = np.random.choice(action_space, p=action_probs.numpy().flatten())

        # 執行行為並獲得回饋
        next_state, reward, done, truncated, info = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)

        # 計算並執行訓練步驟
        action_one_hot = np.zeros(action_space)
        action_one_hot[action] = 1
        loss = train_step(model, state, action_one_hot, reward, next_state, done)

        # 更新狀態
        state = next_state
        episode_reward += reward

    # 記錄回合總回報
    episode_rewards.append(episode_reward)

    # if episode % 50 == 0:
    print(f"Episode {episode}, Reward: {episode_reward}")

# 繪製回報曲線
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.show()

# 測試訓練好的模型
state, _ = env.reset()  # 獲取狀態和額外信息
state = np.expand_dims(state, axis=0)
done = False
steps = 0  # 記錄測試的步數
max_steps = 1000  # 最大步數，可以根據需求調整

while not done and steps < max_steps:
    action_probs, _ = model(state)
    action = np.argmax(action_probs.numpy())  # 根據最大機率選擇行為
    next_state, reward, done, truncated, info = env.step(action)
    env.render()  # 渲染畫面

    state = np.expand_dims(next_state, axis=0)
    steps += 1

# 關閉環境
env.close()

