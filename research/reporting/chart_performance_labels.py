"""Consistent labels from each plotted account's existing net-return metrics."""


def performance_label(name, metrics, multiline=True):
    separator = '\n' if multiline else ' | '
    sharpe = metrics['sharpe']
    displayed_sharpe = 'n/a' if sharpe is None else f'{sharpe:.2f}'
    return f"{name}{separator}CAGR {100*metrics['annualized_return']:.1f}% | SR {displayed_sharpe}"


LEGEND_TITLE = 'Net CAGR | SR = annualized Sharpe\n(3% annual hurdle)'
