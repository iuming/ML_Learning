import matplotlib.pyplot as plt

def plot_temperature(steps, temperatures, target_temperature):
    """
    Plot the temperature over time steps with the target temperature.

    Args:
        steps (list): List of time steps.
        temperatures (list): List of temperatures at each step.
        target_temperature (float): The target temperature to maintain.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(steps, temperatures, label='Temperature')
    plt.axhline(y=target_temperature, color='r', linestyle='--', label='Target Temperature')
    plt.xlabel('Step')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Control Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()