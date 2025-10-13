import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.ticker import FixedLocator


def plot_regression(ax, x, y, data, hue=None, scatter_kws=None, line_kws=None):
    """
    Plot scatterplot with fitted linear regression line.

    Parameters:
    ax (matplotlib.axes.Axes): The axes to plot on.
    x (str): Column name for the x-axis.
    y (str): Column name for the y-axis.
    data (pd.DataFrame): DataFrame containing the data.
    hue (str): Column name for the hue.
    scatter_kws (dict): Keyword arguments for the scatter plot.
    line_kws (dict): Keyword arguments for the regression line.

    Returns:
    None
    """
    if scatter_kws is None:
        scatter_kws = {'s': 50}
    if line_kws is None:
        line_kws = {}

    sns.scatterplot(x=x, y=y, hue=hue, data=data, ax=ax, **scatter_kws)
    if hue:
        for value in data[hue].unique():
            subset = data[data[hue] == value]
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                subset[x], subset[y])
            x_values = np.array([subset[x].min(), subset[x].max()])
            y_values = slope * x_values + intercept
            sns.lineplot(x=x_values, y=y_values, ax=ax,
                         label=f'{value} regression')
    else:
        sns.regplot(x=x, y=y, data=data, ax=ax, ci=None, **line_kws)


def customize_axes(ax, x, y, title, multiple_monkeys=False):
    """
    Customize the axes ticks and labels.

    Parameters:
    ax (matplotlib.axes.Axes): The axes to customize.
    x (str): Column name for the x-axis.
    y (str): Column name for the y-axis.
    x_is_date (bool): Whether the x-axis represents dates.

    Returns:
    None
    """
    # If x-axis is date, only show the year
    if x.lower() == 'date':
        xticks = ax.get_xticks()
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        xticklabels = ax.get_xticklabels()
        xticklabels = [str(label.get_text())[-5:] for label in xticklabels]
        ax.set_xticklabels(xticklabels, rotation=60, ha="right")

    # Make the ylim at least space_between_yticks * 4.5 apart
    space_between_yticks = 0.05
    ymin, ymax = ax.get_ylim()
    if ymax - ymin < space_between_yticks * 3:
        y_avg = (ymin + ymax) / 2
        ax.set_ylim(y_avg - space_between_yticks * 1.5,
                    y_avg + space_between_yticks * 1.5)

    # Make sure the yticks are at least space_between_yticks apart
    if ymax - ymin < space_between_yticks * 10:
        yticks = np.arange(np.floor(ymin / space_between_yticks) * space_between_yticks, np.ceil(
            ymax / space_between_yticks) * space_between_yticks + space_between_yticks, space_between_yticks)
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.set_yticklabels([f"{tick:.2f}" for tick in yticks], fontsize=10)

    if multiple_monkeys:
        ax.legend(prop={'size': 6})

    ax.set_xlabel(x, fontsize=10, loc='left')
    ax.set_ylabel(y, fontsize=10)
    ax.set_title(title, fontsize=12)


def get_title_str(one_pattern_df, x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        one_pattern_df[x], one_pattern_df[y])
    r_squared = r_value ** 2
    num_sessions = one_pattern_df[y].unique().size
    title_str = f"Slope x T = {round(slope * num_sessions, 3)}, R² = {r_squared:.3f}, p = {p_value:.3f}"
    if p_value < 0.05:
        title_str += " *"
    return title_str


def prepare_for_subplots(num_items):
    num_cols = 2
    num_rows = (num_items + num_cols - 1) // num_cols

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
    axes = axes.flatten()
    return fig, axes


def plot_the_changes_over_time_in_long_df(merged_df, x="Data", y='rate', title_column='label', monkey_name='',
                                          multiple_monkeys=False, category_order=None):
    """
    Compare datasets using scatterplots with fitted linear regression lines.

    Parameters:
    merged_df (pd.DataFrame): Merged DataFrame containing the data.
    x (str): Column name for the x-axis.
    y (str): Column name for the y-axis.
    title_column (str): Column name for the title.
    monkey_name (str): Name of the monkey.
    category_order (list): Order of categories to plot.

    Returns:
    None
    """
    sns.set_style("darkgrid")

    if category_order is None:
        category_order = merged_df[title_column].unique()

    fig, axes = prepare_for_subplots(len(category_order))

    for i, item in enumerate(category_order):
        one_pattern_df = merged_df[merged_df['item'] == item]
        ax = axes[i]

        hue = 'monkey' if multiple_monkeys else None
        plot_regression(ax, x, y, one_pattern_df, hue=hue)

        title = get_title(
            one_pattern_df, x, y, one_pattern_df[title_column].iloc[0], multiple_monkeys, monkey_name=monkey_name)
        customize_axes(ax, x, y, title, multiple_monkeys)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_the_changes_over_time_in_wide_df(merged_df, x="Data", y_columns=[], monkey_name='',
                                          multiple_monkeys=False, title_prefix=''):
    """
    Compare datasets using scatterplots with fitted linear regression lines.

    Parameters:
    merged_df (pd.DataFrame): Merged DataFrame containing the data.
    x (str): Column name for the x-axis.
    y (str): Column name for the y-axis.
    title_column (str): Column name for the title.
    monkey_name (str): Name of the monkey.
    category_order (list): Order of categories to plot.

    Returns:
    None
    """

    sns.set_style("darkgrid")

    merged_df = merged_df.copy()

    fig, axes = prepare_for_subplots(len(y_columns))

    for i, y in enumerate(y_columns):

        ax = axes[i]

        # change all '_median' to '_median' in the column name
        if '_median' in y:
            y_old = y
            y = y.replace('_median', '_median')
            merged_df[y] = merged_df[y_old]

        hue = 'monkey' if multiple_monkeys else None
        plot_regression(ax, x, y, merged_df, hue=hue)

        title = get_title(merged_df, x, y, y, multiple_monkeys,
                          monkey_name=monkey_name, title_prefix=title_prefix)
        customize_axes(ax, x, y, title, multiple_monkeys)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def get_title(data_df, x, y, feature_name, multiple_monkeys, monkey_name='', title_prefix=''):
    if multiple_monkeys:
        title = f"{feature_name}\n"
        for monkey in data_df['monkey'].unique():
            monkey_df = data_df[data_df['monkey'] == monkey]
            title_str = get_title_str(monkey_df, x, y)
            title += f"{monkey}: {title_str}\n"
    else:
        title_str = get_title_str(data_df, x, y)
        title = f"{monkey_name}: {feature_name}\n" + title_str

    title = title_prefix + title
    return title
