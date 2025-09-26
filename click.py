import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def collect_clicks_on_image(img_path):
    """Show an image, return a list of (x, y) click positions in pixel coords."""
    img = mpimg.imread(img_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Left-click to mark points. Press Enter to finish.")
    points = []

    # Drawn markers will be stored so we can update them
    scat = ax.scatter([], [], s=30, marker='x')

    def on_click(event):
        # Only register left clicks inside the axes image
        if event.inaxes != ax or event.button != 1:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        points.append((float(x), float(y)))
        # Update scatter with all points so far
        xs, ys = zip(*points)
        scat.set_offsets(list(zip(xs, ys)))
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'enter':
            plt.close(fig)

    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
    cid_key   = fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

    # Clean up connections (good practice if you reuse the figure)
    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)
    return points

# Example:
if __name__ == "__main__":
    pts = collect_clicks_on_image("004.jpg")
    print("Clicked points (x, y):", pts)
