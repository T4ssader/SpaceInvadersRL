import tkinter as tk

class QLearningGUI:
    def __init__(self, game, agent):
        self.root = tk.Tk()
        self.root.title("QLearning Control Panel")
        self.game = game
        self.agent = agent
        self.epsilon = agent.epsilon
        self.gamma = agent.gamma
        self.alpha = agent.alpha
        self.create_widgets()
        self.steps_to_execute = 0

    def create_widgets(self):
        # Create Epsilon label and scale widget
        epsilon_label = tk.Label(self.root, text="Epsilon")
        epsilon_label.grid(row=0, column=0)
        epsilon_scale = tk.Scale(self.root, from_=0, to=1, resolution=0.1,
                                 orient=tk.HORIZONTAL, command=self.set_epsilon)
        epsilon_scale.set(self.epsilon)
        epsilon_scale.grid(row=0, column=1)

        # Create Gamma label and scale widget
        gamma_label = tk.Label(self.root, text="Gamma")
        gamma_label.grid(row=1, column=0)
        gamma_scale = tk.Scale(self.root, from_=0, to=1, resolution=0.1,
                               orient=tk.HORIZONTAL, command=self.set_gamma)
        gamma_scale.set(self.gamma)
        gamma_scale.grid(row=1, column=1)

        # Create Alpha label and scale widget
        alpha_label = tk.Label(self.root, text="Alpha")
        alpha_label.grid(row=2, column=0)
        alpha_scale = tk.Scale(self.root, from_=0, to=1, resolution=0.1,
                               orient=tk.HORIZONTAL, command=self.set_alpha)
        alpha_scale.set(self.alpha)
        alpha_scale.grid(row=2, column=1)

        # Create Next Step button
        next_step_button = tk.Button(self.root, text="Next Step", command=self.next_step)
        next_step_button.grid(row=3, column=0)

        # Create Next Episode button
        next_episode_button = tk.Button(self.root, text="Next Episode", command=self.next_episode)
        next_episode_button.grid(row=3, column=1)

        # Create Print Q-Table button
        print_q_table_button = tk.Button(self.root, text="Print Q-Table", command=self.print_q_table)
        print_q_table_button.grid(row=4, column=0)

        # Create Print Best Action button
        print_best_action_button = tk.Button(self.root, text="Print Best Action", command=self.print_best_action)
        print_best_action_button.grid(row=4, column=1)

        # Method to print Q-Table
    def print_q_table(self):
        print(self.agent.q_table)

    def set_epsilon(self, val):
        self.epsilon = float(val)
        self.agent.set_epsilon(self.epsilon)

    # Method to print Best Action
    def print_best_action(self):
        prtint("Best ACtion?")

    def set_gamma(self, val):
        self.gamma = float(val)
        self.agent.set_gamma(self.gamma)

    def set_alpha(self, val):
        self.alpha = float(val)
        self.agent.set_alpha(self.alpha)

    def next_step(self):
        self.steps_to_execute = 1
    def next_episode(self):
        self.steps_to_execute = 2147483647

    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    gui = QLearningGUI()
    gui.run()
