import requests
from bs4 import BeautifulSoup
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
from matplotlib.widgets import CheckButtons
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ttkbootstrap as ttk
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import norm
from datetime import datetime
import math
from pathlib import Path
import ctypes
import yfinance as yf
from tqdm import tqdm
import clarabel
import scipy


def get_website(url):
    response = requests.get(url)
    if not response.ok:
        print('Status code:', response.status_code)
        raise Exception('Failed to load page {}'.format(url))
    page_content = response.text
    doc = BeautifulSoup(page_content, 'html.parser')
    return doc


class YieldCurve:
    def __init__(self, locations, function_indices, links):
        print('Setting up yield curves...')
        self.locations = locations
        self.functions = [self.get_wgb_rate]
        self.function_indices = function_indices
        self.links = links
        self.yield_data = [[[], [], []] for _ in range(len(locations))]
        for num, location in enumerate(self.locations):
            yield_data_path = Path(location + '_yield_data.txt')
            try:
                yield_data_path.resolve(strict=True)
            except FileNotFoundError:
                yield_text_file = open(location + '_yield_data.txt', 'x')
                self.refresh_yield_data(location, yield_text_file, self.function_indices[num])
            else:
                yield_text_file = open(location + '_yield_data.txt', 'r+')
                lines = yield_text_file.readlines()
                date = datetime.today()
                if (len(lines) == 0 or lines[-4].rstrip().split('-')[:3] !=
                        [date.strftime('%Y'), date.strftime('%m'), date.strftime('%d')]):
                    yield_text_file.seek(0)
                    yield_text_file.truncate()
                    self.refresh_yield_data(location, yield_text_file, self.function_indices[num])
                else:
                    self.refresh_date = datetime(*[int(elem) for elem in lines[-4].rstrip().split('-')])
                    self.yield_data[self.locations.index(location)][0] = lines[-3].rstrip().split(',')
                    self.yield_data[self.locations.index(location)][1] = [float(elem) for elem in
                                                                          lines[-2].rstrip().split(',')]
                    self.yield_data[self.locations.index(location)][2] = [float(elem) for elem in
                                                                          lines[-1].rstrip().split(',')]
                    yield_text_file.close()

    def get_wgb_rate(self, link):
        page = get_website(link)
        table = page.find(['table'], {'class': ['w3-table money pd22 -f14']})
        table = table.findChildren('tbody', recursive=False)[0]
        self.yield_data[self.links.index(link)] = [[], [], []]
        for row in table.findChildren('tr'):
            entries = row.findChildren('td')
            maturity = entries[1].text.strip()
            self.yield_data[self.links.index(link)][0].append(maturity)
            maturity = maturity.split(' ')
            if maturity[1] in ['month', 'months']:
                self.yield_data[self.links.index(link)][1].append(round(float(maturity[0]) / 12, 3))
            elif maturity[1] in ['year', 'years']:
                self.yield_data[self.links.index(link)][1].append(float(maturity[0]))
            self.yield_data[self.links.index(link)][2].append(float(entries[2].text.strip()[:-1]))

    def return_yield_data(self):
        return self.yield_data

    def refresh_yield_data(self, location, yield_data, function_index):
        self.functions[function_index](self.links[self.locations.index(location)])
        date = datetime.today()
        yield_data.write(date.strftime('%Y-%m-%d-%H-%M-%S') + '\n')
        self.refresh_date = date
        for j in range(3):
            yield_data.write(
                ','.join([str(item) for item in self.yield_data[self.locations.index(location)][j]]) + '\n')
        yield_data.close()

    def refresh(self):
        for num, location in enumerate(self.locations):
            yield_text_file = open(location + '_yield_data.txt', 'w')
            self.refresh_yield_data(location, yield_text_file, self.function_indices[num])

    def return_refresh_date(self):
        return self.refresh_date


class InterestRateProcess:
    def __init__(self):
        print('Interpolating yield curve...')
        self.us_data = [[], []]
        self.yield_spline = None
        us_spot_data_path = Path('US_data.txt')
        try:
            us_spot_data_path.resolve(strict=True)
        except FileNotFoundError:
            us_data_file = open('US_data.txt', 'x')
            self.refresh_spot_data(us_data_file)
        else:
            us_data_file = open('US_data.txt', 'r+')
            lines = us_data_file.readlines()
            date = datetime.today()
            if (len(lines) == 0 or lines[-3].rstrip().split('-')[:3] !=
                    [date.strftime('%Y'), date.strftime('%m'), date.strftime('%d')]):
                us_data_file.seek(0)
                us_data_file.truncate()
                self.refresh_spot_data(us_data_file)
            else:
                self.refresh_date = datetime(*[int(elem) for elem in lines[-3].rstrip().split('-')])
                self.us_data[0] = [float(elem) for elem in lines[-2].rstrip().split(',')]
                self.us_data[1] = [float(elem) for elem in lines[-1].rstrip().split(',')]
                self.yield_spline = interpolate.CubicSpline(self.us_data[0], self.us_data[1])
                us_data_file.close()

    def get_cnbc_rate(self):
        self.us_data = [[], []]
        with tqdm(total=13) as progress_bar:
            for maturity in ['1M', '2M', '3M', '4M', '6M', '1Y']:
                page = get_website('https://www.cnbc.com/quotes/US' + maturity)
                current_yield = float(page.find(['span'], {'class': ['QuoteStrip-lastPrice']}).text.strip('%'))
                expiry = page.find(['li'], {'class': ['Summary-stat Summary-maturity']})
                expiry = expiry.findChildren('span', recursive=False)[1].text
                expiry = datetime.strptime(expiry, '%Y-%m-%d')
                time_to_expiry = expiry - datetime.today()
                duration = round((time_to_expiry.days + time_to_expiry.seconds / 86400) / 365.2425, 3)
                self.us_data[0].append(duration)
                self.us_data[1].append(current_yield)
                progress_bar.update(1)
            for maturity in ['2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']:
                page = get_website('https://www.cnbc.com/quotes/US' + maturity)
                current_yield = float(page.find(['span'], {'class': ['QuoteStrip-lastPrice']}).text.strip('%'))
                expiry = page.find(['li'], {'class': ['Summary-stat Summary-maturity']})
                expiry = expiry.findChildren('span', recursive=False)[1].text
                expiry = datetime.strptime(expiry, '%Y-%m-%d')
                coupon = page.find(['li'], {'class': ['Summary-stat Summary-coupon']})
                coupon = float(coupon.findChildren('span', recursive=False)[1].text.strip('%'))
                time_to_expiry = expiry - datetime.today()
                years_to_expiry = round((time_to_expiry.days + time_to_expiry.seconds / 86400) / 365.2425, 3)
                tot = 0
                for dates in [years_to_expiry - 0.5 * i for i in range(math.floor(years_to_expiry * 2) + 1)]:
                    tot += coupon * dates / 2
                tot += 100 * years_to_expiry
                duration = tot / (100 + (math.floor(years_to_expiry * 2) + 1) * coupon / 2)
                self.us_data[0].append(duration)
                self.us_data[1].append(current_yield)
                self.yield_spline = interpolate.CubicSpline(self.us_data[0], self.us_data[1])
                progress_bar.update(1)

    def return_us_data(self):
        return self.us_data

    def refresh_spot_data(self, us_data):
        self.get_cnbc_rate()
        date = datetime.today()
        us_data.write(date.strftime('%Y-%m-%d-%H-%M-%S') + '\n')
        self.refresh_date = date
        for j in range(2):
            us_data.write(','.join([str(item) for item in self.us_data[j]]) + '\n')
        us_data.close()

    def refresh(self):
        us_data_file = open('US_data.txt', 'w')
        self.refresh_spot_data(us_data_file)

    def return_refresh_date(self):
        return self.refresh_date

    def get_yield_spline(self):
        return self.yield_spline


class OptionsData:
    def __init__(self, ticker: str, load=False):
        print('Getting option data...')
        self.spx = yf.Ticker(ticker)
        self.spot = None
        self.expirations = None
        self.expiration_years = None
        self.options = None
        self.refresh_date = None
        self.forward = []
        self.refresh(load)

    def compute_iv(self, df, market):
        def option_price_vega(option_type: str, S, K, T, r, sigma):
            d1 = math.log(S/K) / (sigma*math.sqrt(T)) + (r/sigma + sigma/2) * math.sqrt(T)
            d2 = d1 - sigma * math.sqrt(T)
            vega = S * math.sqrt(T) * norm.pdf(d1)
            if option_type == 'Call':
                price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            elif option_type == 'Put':
                price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            else:
                return None
            return [price, vega]

        def newton_raphson(row, price_type='mid'):
            target_error = 10 ** (-5)
            r = math.log(row['forward']/self.spot)/row['T']
            guess = math.sqrt(abs(2 * r + 2 * math.log(self.spot / row['strike']) / row['T']))
            if row['type'] in ['Call', 'Put']:
                if price_type == 'mid':
                    target = (row['bid'] + row['ask'])/2
                else:
                    target = row[price_type]
                try:
                    while True:
                        [price, vega] = option_price_vega(row['type'], self.spot, row['strike'], row['T'], r, guess)
                        if abs(price - target) <= target_error:
                            return guess
                        else:
                            guess -= (price - target) / vega
                except ZeroDivisionError:
                    return None
            else:
                return None
        if market:
            df.loc[:, 'VolImp'] = df.apply(lambda x: newton_raphson(x, 'mid'), axis=1)
        else:
            df.loc[:, 'SmoothVolImp'] = df.apply(lambda x: newton_raphson(x, 'smoothed'), axis=1)
        return df

    def return_options_data(self):
        return self.options

    def return_options_expirations(self):
        return self.expirations, self.expiration_years

    def return_refresh_date(self):
        return self.refresh_date
    
    def return_forward(self):
        return self.forward

    def refresh(self, load):
        self.options = []
        options_expirations_path = Path('options/expirations.txt')
        try:
            options_expirations_path.resolve(strict=True)
        except FileNotFoundError:
            expiration_file = open('options/expirations.txt', 'x')
            file = False
        else:
            expiration_file = open('options/expirations.txt', 'r+')
            file = True

        if load and file:
            expiration_file = open('options/expirations.txt', 'r+')
            lines = expiration_file.readlines()
            self.expirations = lines[0].split(',')
            self.expiration_years = [float(elem) for elem in lines[1].split(',')]
            self.refresh_date = datetime(*[int(elem) for elem in lines[2].split('-')])
            self.spot = float(lines[3])
            self.forward = [float(elem) for elem in lines[4].split(',')]
            expiration_file.close()
            for expiration in self.expirations:
                df = pd.read_pickle('options/' + expiration.strip())
                self.options.append(df)
            print('Local options data loaded')
            return

        expiration_file.seek(0)
        expiration_file.truncate()
        self.expiration_years = []
        #if 'ask' in self.spx.info.keys() and 'bid' in self.spx.info.keys():
        #    self.spot = (self.spx.info['bid'] + self.spx.info['ask']) / 2
        #else:
        self.spot = self.spx.history()['Close'].iloc[-1]
        self.expirations = list(self.spx.options)
        temp = []
        def put_to_call(row):
            if row['type'] == 'Call':
                return (row['bid'] + row['ask']) / 2
            else:
                return (row['bid'] + row['ask']) / 2 + self.spot - row['strike'] * self.spot/row['forward']
        with tqdm(total=len(self.expirations)) as progress_bar:
            for expiration in self.expirations:
                current = self.spx.option_chain(expiration)
                calls = current.calls
                calls.insert(0, 'type', 'Call')
                puts = current.puts
                puts.insert(0, 'type', 'Put')
                df = pd.concat([calls, puts], ignore_index=True)
                df['lastTradeDate'] = df['lastTradeDate'].apply(lambda x: x.replace(tzinfo=None))
                df = df[(df['lastTradeDate'].apply(lambda x: (datetime.today()-x).days <= 3)) & df['volume'] > 0]
                df = df[(df['bid'] != 0) & (df['ask'] != 0)]
                df.drop(columns=['change', 'percentChange', 'volume', 'impliedVolatility',
                                 'inTheMoney', 'contractSize', 'currency', 'openInterest'], inplace=True)
                # SPX options typically settle at 4pm on expiry date
                time_to_expiry = (datetime(*([int(elem) for elem in expiration.split('-')] + [16, 0, 0])) -
                                  datetime.today())
                # 5 hours difference between time zones
                Ts = [(time_to_expiry.days + 5/24 - 6.5/24 + time_to_expiry.seconds/86400) / 365.2425,
                      (time_to_expiry.days + 5/24 + time_to_expiry.seconds / 86400) / 365.2425]
                if len(df.index) == 0:
                    progress_bar.update(1)
                    continue
                weekly = df[df['contractSymbol'].str.contains('W')]
                monthly = df[~df['contractSymbol'].str.contains('W')]
                for i, df in enumerate([monthly, weekly]):
                    if len(df.index) == 0:
                        continue
                    T = Ts[i]
                    # 0.003 years = 1 day
                    if T > 0.003 and len(df[df['type'] == 'Call'].index) >= 2 and len(df[df['type'] == 'Put'].index) >= 2:
                        df['T'] = T
                        # Estimate forward price using intersections
                        c, p = df[df['type'] == 'Call'], df[df['type'] == 'Put']
                        l = max(c['strike'].min(), p['strike'].min())
                        r = min(c['strike'].max(), p['strike'].max())
                        xs = np.linspace(l, r, 2500)
                        i1, i2 = 0, 0
                        pa, pb, ca, cb = np.array([]), np.array([]), np.array([]), np.array([])
                        for x in xs:
                            while i1+2 < len(c.index) and x > c['strike'].iloc[i1+1]:
                                i1 += 1
                            while i2+2 < len(p.index) and x > p['strike'].iloc[i2+1]:
                                i2 += 1
                            pa = np.append(pa, p['ask'].iloc[i2] + (x-p['strike'].iloc[i2])*
                                           (p['ask'].iloc[i2+1]-p['ask'].iloc[i2])/(p['strike'].iloc[i2+1]-p['strike'].iloc[i2]))
                            pb = np.append(pb, p['bid'].iloc[i2] + (x-p['strike'].iloc[i2])*
                                           (p['bid'].iloc[i2+1]-p['bid'].iloc[i2])/(p['strike'].iloc[i2+1]-p['strike'].iloc[i2]))
                            ca = np.append(ca, c['ask'].iloc[i1] + (x-c['strike'].iloc[i1])*
                                           (c['ask'].iloc[i1+1]-c['ask'].iloc[i1])/(c['strike'].iloc[i1+1]-c['strike'].iloc[i1]))
                            cb = np.append(cb, c['bid'].iloc[i1] + (x-c['strike'].iloc[i1])*
                                           (c['bid'].iloc[i1+1]-c['bid'].iloc[i1])/(c['strike'].iloc[i1+1]-c['strike'].iloc[i1]))
                        idx1 = np.argwhere(np.diff(np.sign(pa-cb))).flatten()
                        idx2 = np.argwhere(np.diff(np.sign(pb-ca))).flatten()
                        if len(idx1) > 0 and len(idx2) > 0:
                            forward_price = (xs[idx1[0]] + xs[idx2[0]])/2
                            if len(self.forward) == 0 or forward_price >= self.forward[-1]:
                                df['forward'] = forward_price
                                df['logMoneyness'] = np.log(df['strike'] / forward_price)
                                df = df[(df['logMoneyness'] > -0.8) & (df['logMoneyness'] < 0.4)]
                                df = df[((df['logMoneyness'] <= 0) & (df['type'] == 'Put')) |
                                        ((df['logMoneyness'] > 0) & (df['type'] == 'Call'))]
                                df = self.compute_iv(df, True).sort_values(by=['logMoneyness'], ignore_index=True)
                                df['CallPrice'] = df.apply(put_to_call, axis=1)
                                if len(df.index) >= 5:
                                    self.forward.append(forward_price)
                                    temp.append(expiration + ('M' if i == 0 else 'W'))
                                    self.expiration_years.append(T)
                                    self.options.append(df)
                                    df.to_pickle('options/' + expiration + ('M' if i == 0 else 'W'))
                progress_bar.update(1)

        print(str(len(self.expirations)-len(self.options)) + ' maturities dropped')
        self.expirations = temp
        self.refresh_date = datetime.today()
        expiration_file.write(','.join(self.expirations) + '\n')
        expiration_file.write(','.join([str(elem) for elem in self.expiration_years]) + '\n')
        expiration_file.write(self.refresh_date.strftime('%Y-%m-%d-%H-%M-%S') + '\n')
        expiration_file.write(str(self.spot) + '\n')
        expiration_file.write(','.join([str(elem) for elem in self.forward]))
        expiration_file.close()

    def return_spot(self):
        return self.spot


class SmoothPrices:
    def __init__(self, dfs, spot):
        print('Smoothing option prices...')
        self.dfs = dfs[::-1]
        self.g, self.second, self.h = [], [], []
        self.spot = spot
        non_convex = 0
        for i, df in enumerate(self.dfs):
            df['strike2'] = df['strike'].shift(1)
            g, second, h = self.smooth_prices(df, i)
            self.g.append(g)
            self.second.append(second)
            self.h.append(h)
            if min(second) < 0:
                non_convex += 1
        print(str(non_convex) + ' non-convex curves out of ' + str(len(self.dfs)) + ' maturities')

    def return_smoothed_price(self, index, xs):
        def price(index, x, start):
            strikes = self.dfs[index]['strike'].iloc[1:-1]
            i = start
            g, second, h = self.g[index], self.second[index], self.h[index]
            if x < strikes.iloc[0] or x > strikes.iloc[-1]:
                return None, i
            while x > strikes.iloc[i + 1]:
                i += 1
            prev, after = strikes.iloc[i], strikes.iloc[i + 1]
            y = (((x-prev) * g[i+1] + (after-x) * g[i]) / h[i] -
                 (1/6) * (x-prev) * (after-x) * (
                         (1 + (x-prev)/h[i])*second[i + 1] + (1+(after-x)/h[i])*second[i]))
            return y, i

        i = 0
        ys = []
        for x in xs:
            y, interval = price(index, x, i)
            ys.append(y)
            i = interval
        return ys

    def return_rnd(self, index, xs):
        def convexity(index, x, start):
            strikes = self.dfs[index]['strike'].iloc[1:-1]
            i = start
            g, second, h = self.g[index], self.second[index], self.h[index]
            if x < strikes.iloc[0] or x > strikes.iloc[-1]:
                return None, i
            while x > strikes.iloc[i + 1]:
                i += 1
            prev, after = strikes.iloc[i], strikes.iloc[i + 1]
            y = (((x-prev) * g[i+1] + (after-x) * g[i]) / h[i] -
                 (1/6) * (x-prev) * (after-x) * (
                         (1 + (x-prev)/h[i])*second[i + 1] + (1+(after-x)/h[i])*second[i]))
            return y, i

        i = 0
        ys = []
        for x in xs:
            y, interval = convexity(index, x, i)
            ys.append(y)
            i = interval
        return ys

    def smooth_prices(self, df, i):
        n = len(df.index) - 2
        h = np.array(df['strike'] - df['strike2'])[2:-1]  # 1 to n-1
        Q = np.vstack((np.diag(1 / h[:-1]),
                       np.zeros((2, n-2)))) + np.vstack((np.zeros((1, n - 2)),
                                              -np.diag(1/h[:-1]) - np.diag(1/h[1:]),
                                              np.zeros((1, n - 2)))) + np.vstack((np.zeros((2, n - 2)),
                                                                                  np.diag(1 / h[1:])))
        R = np.diag((1/3)*h[:-1] + (1/3)*h[1:]) + np.diag((1/6)*h[1:-1], k=1) + np.diag((1/6)*h[1:-1], k=-1)
        A = np.hstack((np.transpose(Q), -R))
        B = np.vstack((np.hstack((np.eye(n), np.zeros((n, n - 2)))),
                       np.hstack((np.zeros((n-2, n)), R))))
        y = np.hstack((np.array(df['CallPrice'].iloc[1:n+1]),
                       np.zeros(n-2)))
        T = df['T'].iloc[0]
        G = np.vstack((A,
                       np.concatenate((np.zeros((n-2, n)), -np.eye(n-2)), axis=1),
                       np.concatenate(([[1/h[0], -1/h[0]]], np.zeros((1, n-2)), [[h[0]/6]],
                                       np.zeros((1, n-3))), axis=1),
                       np.concatenate((np.zeros((1, n-2)), [[-1/h[-1], 1/h[-1]]],
                                       np.zeros((1, n-3)), [[h[-1]/6]]), axis=1),
                       np.concatenate(([[-1]], np.zeros((1, 2*n-3))), axis=1),
                       np.concatenate((np.zeros((1, n-1)), [[-1]], np.zeros((1, n-2))), axis=1)))
        l = np.hstack((np.zeros(n-2),
                       np.zeros(n-2),
                       [self.spot/df['forward'].iloc[0],
                        0,
                        -self.spot + df['strike'].iloc[1] * self.spot/df['forward'].iloc[0],
                        -0.01]))
        if i == 0:
            G = np.vstack((G,
                           np.concatenate(([[1]], np.zeros((1, 2*n-3))), axis=1)))
            l = np.hstack((l,
                           [self.spot]))
            cones = [clarabel.ZeroConeT(n-2),
                     clarabel.NonnegativeConeT(n+3)]
        else:
            rate = self.dfs[i-1]['forward'].iloc[0]/df['forward'].iloc[0]
            G = np.vstack((G,
                           np.hstack((np.eye(n), np.zeros((n, n-2))))))
            cones = [clarabel.ZeroConeT(n-2),
                     clarabel.NonnegativeConeT(2*n+2)]
            def temp(x):
                return self.spot if x is None else x
            prices = np.array([temp(elem) for elem in self.return_smoothed_price(i-1, rate*df['strike'].iloc[1:-1])])
            l = np.hstack((l,
                           prices))
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        solver = clarabel.DefaultSolver(scipy.sparse.csc_matrix(B), -y,
                                        scipy.sparse.csc_matrix(G), l, cones, settings)
        sol = solver.solve()
        return sol.x[:n], np.hstack((np.zeros(1), sol.x[n:], np.zeros(1))), h


def rescale_figure_range(ax, vis_only=True):
    ax.relim(visible_only=vis_only)
    ax.autoscale_view()


def toggle_plot(label):
    index = locations.index(label)
    yield_lines[index].set_visible(not yield_lines[index].get_visible())
    rescale_figure_range(main_ax1)
    fig1.canvas.draw_idle()


def update_annot(line_data, ind, loc):
    maturities, x, y = line_data
    annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
    text = loc + f' {maturities[ind["ind"][0]]}, {y[ind["ind"][0]]}%'
    point_annot.set_data([x[ind["ind"][0]]], [y[ind["ind"][0]]])
    point_annot.set_visible(True)
    annot.set_text(text)
    annot.set_visible(True)
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event, lines, data):
    vis = annot.get_visible()
    if event.inaxes == main_ax1:
        for index, line in enumerate(lines):
            if line.get_visible():
                contains_point, point_index = line.contains(event)
                if contains_point:
                    update_annot(data[index], point_index, locations[index])
                    fig1.canvas.draw_idle()
                    return
        if vis:
            annot.set_visible(False)
            point_annot.set_visible(False)
            fig1.canvas.draw_idle()


def refresh_yield(lines):
    yield_curve.refresh()
    new_data = yield_curve.return_yield_data()
    for i, line in enumerate(lines):
        line.set_xdata(new_data[i][1])
        line.set_ydata(new_data[i][2])
        if not line.get_visible():
            plot_toggle_button.set_active(i)
    rescale_figure_range(main_ax1)
    refresh1.config(text=yield_curve.return_refresh_date().strftime('Last updated \n' + '%Y-%m-%d %H:%M:%S'))


def refresh_interest(lines):
    interest_process_US.refresh()
    new_data = interest_process_US.return_us_data()
    lines[0].set_xdata(new_data[0])
    lines[0].set_ydata(new_data[1])
    rescale_figure_range(main_ax2)
    refresh2.config(text=interest_process_US.return_refresh_date().strftime('Last updated \n' + '%Y-%m-%d %H:%M:%S'))
    refresh_options()


def refresh_options():
    global lines
    for line in lines:
        for l in line:
            l.remove()
    options_data.refresh(load=False)
    new_data = options_data.return_options_data()
    expirations, expiration_years = options_data.return_options_expirations()
    global spot, smoother, smoothed_dfs
    spot = options_data.return_spot()
    smoother = SmoothPrices(new_data, spot)
    smoothed_dfs, non_converging, total = return_smoothed_dfs(smoother, N=100)
    print(str(non_converging) + ' non-converging solutions out of ' + str(total) + ' data points')

    l1 = [main_ax3.scatter(new_data[i]['strike'], new_data[i]['CallPrice'], c='yellow',
                           visible=bool(i == 0), s=4, marker='x') for i in range(len(expirations))]
    l2 = [main_ax3.plot(smoothed_dfs[i]['strike'], smoothed_dfs[i]['smoothed'], c='red',
                        linewidth=1.5, visible=bool(i == 0))[0]
          for i in range(len(expirations))]
    l3 = [main_ax3.axvline(x=forward[i], linestyle='--',
                           visible=bool(i == 0)) for i, T in enumerate(expiration_years)]
    lines = [l1, l2, l3]
    global white_patch, price_vol, strike_moneyness
    price_vol, strike_moneyness = 1, 1
    price_vol_switch_button.config(image=slider_off1)
    strike_moneyness_switch_button.config(image=slider_off2)
    white_patch = mpl_patches.Patch(color='white', label='Forward Price')
    main_ax3.legend(handles=[yellow_patch, red_patch, white_patch])
    global menu
    menu.configure(values=expirations)
    menu.set(expirations[0])

    rescale_figure_range(main_ax3, False)
    fig3.canvas.draw_idle()
    x = np.concatenate([new_data[i]['strike'] for i in range(len(new_data))])
    y = np.concatenate([np.repeat(math.log(expiration_years[i]),
                                  len(new_data[i].index)) for i in range(len(new_data))])
    z = np.concatenate([new_data[i]['VolImp'] for i in range(len(new_data))])
    points = np.array([x, y]).T
    x_grid, y_grid = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    z_grid = interpolate.griddata(points, z, (x_grid, y_grid), method='linear')
    global surf
    surf.remove()
    surf = main_ax4.plot_surface(x_grid, y_grid, z_grid, rstride=1, cstride=1, cmap=cm.hsv)
    main_ax4.view_init(20, 230)
    fig4.canvas.draw_idle()
    refresh3.config(text=options_data.return_refresh_date().strftime('Last updated \n' + '%Y-%m-%d %H:%M:%S'))


def change_maturity(lines):
    global current_index
    maturity = menu.get()
    index = menu['values'].index(maturity)
    for line in lines:
        line[current_index].set_visible(False)
        line[index].set_visible(True)
    fig3.canvas.draw_idle()
    current_index = index


def return_smoothed_dfs(smoother, N):
    smoothed_dfs = []
    non_converging, total = 0, 0
    with tqdm(total=len(smoother.dfs)) as progress_bar:
        for i, df in enumerate(smoother.dfs[::-1]):
            xs = np.linspace(df['strike'].min(), df['strike'].max(), N)
            ys = smoother.return_smoothed_price(len(smoother.dfs)-1-i, xs)
            T = df['T'].iloc[0]
            forward = df['forward'].iloc[0]
            smoothed_df = pd.DataFrame(data={'strike': xs,
                                             'smoothed': ys,
                                             'T': T,
                                             'forward': forward,
                                             'logMoneyness': np.log(xs / forward),
                                             'type': 'Call'}).dropna(subset=['smoothed'])
            smoothed_df = options_data.compute_iv(smoothed_df, False)
            non_converging += smoothed_df[smoothed_df['SmoothVolImp'] == None].shape[0]
            total += smoothed_df.shape[0]
            smoothed_dfs.append(smoothed_df)
            progress_bar.update(1)
    return smoothed_dfs, non_converging, total


def price_vol_switch():
    global lines
    l1, l2, l3 = lines
    for l in l1:
        l.remove()
    global price_vol, strike_moneyness
    x_key = 'strike' if strike_moneyness == 1 else 'logMoneyness'
    if price_vol == 1:
        price_vol_switch_button.config(image=slider_on1)
        main_ax3.set_title('SPX Options Implied Volatility', y=1.05, fontsize=16)
        y_key1 = 'VolImp'
        y_key2 = 'SmoothVolImp'
    else:
        price_vol_switch_button.config(image=slider_off1)
        main_ax3.set_title('SPX Options Prices', y=1.05, fontsize=16)
        y_key1 = 'CallPrice'
        y_key2 = 'smoothed'
    new_l1 = [main_ax3.scatter(options_df[i][x_key], options_df[i][y_key1], c='yellow',
                               visible=bool(i == current_index), s=4, marker='x')
              for i in range(len(options_expirations))]
    for i, l in enumerate(l2):
        l.set_ydata(smoothed_dfs[i][y_key2])
    lines = [new_l1, l2, l3]
    rescale_figure_range(main_ax3, False)
    price_vol = -price_vol
    fig3.canvas.draw_idle()


def strike_moneyness_switch():
    global lines
    l1, l2, l3 = lines
    for l in l1:
        l.remove()
    global strike_moneyness, price_vol
    y_key = 'CallPrice' if price_vol == 1 else 'VolImp'
    if strike_moneyness == 1:
        strike_moneyness_switch_button.config(image=slider_on2)
        x_key = 'logMoneyness'
        for i, l in enumerate(l3):
            l.set_xdata([0])
        white_patch.set_label('Forward Log-Moneyness')
    else:
        strike_moneyness_switch_button.config(image=slider_off2)
        x_key = 'strike'
        for i, l in enumerate(l3):
            l.set_xdata([forward[i]])
        white_patch.set_label('Forward Price')
    main_ax3.legend(handles=[yellow_patch, red_patch, white_patch])
    new_l1 = [main_ax3.scatter(options_df[i][x_key], options_df[i][y_key], c='yellow',
                                             visible=bool(i == current_index), s=4, marker='x')
              for i in range(len(options_expirations))]
    for i, l in enumerate(l2):
        l.set_xdata(smoothed_dfs[i][x_key])
    lines = [new_l1, l2, l3]
    rescale_figure_range(main_ax3, False)
    strike_moneyness = -strike_moneyness
    fig3.canvas.draw_idle()


locations = ['HK', 'US', 'UK']
retrieve_function_indices = [0, 0, 0]
retrieve_functions_links = ['https://www.worldgovernmentbonds.com/country/hong-kong/#title-curve',
                            'https://www.worldgovernmentbonds.com/country/united-states/#title-curve',
                            'https://www.worldgovernmentbonds.com/country/united-kingdom/#title-curve']
yield_curve = YieldCurve(locations,
                         retrieve_function_indices,
                         retrieve_functions_links)
interest_process_US = InterestRateProcess()

# Create window
ctypes.windll.shcore.SetProcessDpiAwareness(2)
window = ttk.Window()
window.configure(background='black')
window.geometry('2560x1440')
window.attributes('-fullscreen', True)
# Configuring grid
window.grid_columnconfigure(0, weight=20, uniform='window')
window.grid_columnconfigure(1, weight=19, uniform='window')
window.grid_columnconfigure(2, weight=1, uniform='window')
window.grid_rowconfigure(0, weight=1, uniform='window')
window.grid_rowconfigure(1, weight=12, uniform='window')
window.grid_rowconfigure(2, weight=12, uniform='window')
# Creating frames
s = ttk.Style()
s.configure('Grey.TFrame', background='grey')
s.configure('WhiteButton.TLabel', background='white')
title_frame = ttk.Frame(master=window, style='Grey.TFrame')
quit_frame = ttk.Frame(master=window, style='Grey.TFrame')
frame1 = ttk.Frame(master=window, style='Grey.TFrame')
frame2 = ttk.Frame(master=window, style='Grey.TFrame')
frame3 = ttk.Frame(master=window, style='Grey.TFrame')
frame4 = ttk.Frame(master=window, style='Grey.TFrame')
# Placing frames
title_frame.grid(row=0, column=0, columnspan=2, sticky='nswe', padx=(5, 0), pady=(5, 0))
quit_frame.grid(row=0, column=2, sticky='nswe', padx=(10, 0), pady=(5, 0))
frame1.grid(row=1, column=0, padx=5, pady=5, sticky='nswe')
frame2.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky='nswe')
frame3.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky='nswe')

plt.style.use('dark_background')
refresh_icon = ttk.PhotoImage(file=r'./png/refresh.png')
refresh_icon = refresh_icon.subsample(13, 13)
close_icon = ttk.PhotoImage(file=r'./png/close.png')
close_icon = close_icon.subsample(5, 5)
project_title = ttk.Label(master=title_frame, text='Volatility Surface Project',
                          foreground='black', font=("Courier", 20))
project_title.place(relx=0.003, rely=0.05, relwidth=0.994)
close_button = ttk.Button(master=quit_frame, image=close_icon, style='WhiteButton.TLabel', command=window.destroy)
close_button.place(relx=0, rely=0, relwidth=1, relheight=1)
# Window 1 (yields)
fig1, button_ax1 = plt.subplots()
button_ax1.set_position((0.85, 0.4, 0.13, 0.2))
main_ax1 = fig1.add_axes((0.08, 0.18, 0.75, 0.65))
canvas1 = FigureCanvasTkAgg(fig1, master=frame1)
canvas1.get_tk_widget().place(relx=0.005, rely=0.01, relwidth=0.99, relheight=0.98)
refresh1 = ttk.Label(master=frame1,
                     text=yield_curve.return_refresh_date().strftime('Last updated \n' + '%Y-%m-%d %H:%M:%S'),
                     foreground='white', background='black', justify='center')
refresh1.place(relx=0.815, rely=0.87, relwidth=0.18, relheight=0.12)

yield_data = yield_curve.return_yield_data()
line_colors = ['green', 'red', 'blue']
yield_lines = [main_ax1.plot(yield_data[i][1], yield_data[i][2], c=line_colors[i],
                             label=locations[i], visible=bool(i == 0))[0] for i in range(len(locations))]

# Annotation and dot on hover
annot = main_ax1.annotate("", xy=(0, 0), xytext=(-10, 10), textcoords='offset points',
                          bbox=dict(boxstyle="round", fc="w"))
annot.set_visible(False)
point_annot = plt.plot(0, 0, c='black', marker='o', markersize=3)[0]
point_annot.set_visible(False)
fig1.canvas.mpl_connect("motion_notify_event", lambda event: hover(event, yield_lines, yield_data))
# Toggle locations
button_colors = ['green', 'red', 'blue']
plot_toggle_button = CheckButtons(ax=button_ax1, labels=locations, actives=[True, False, False],
                                  label_props={'color': button_colors},
                                  frame_props={'edgecolor': button_colors},
                                  check_props={'facecolor': button_colors})
for i in range(len(button_colors)):
    label = plot_toggle_button.labels[i]
    label.set_x(label.get_position()[0] + 0.15)
plot_toggle_button.on_clicked(toggle_plot)

refresh_button1 = ttk.Button(master=frame1, image=refresh_icon, style='WhiteButton.TLabel',
                             command=lambda: refresh_yield(yield_lines))
refresh_button1.place(relx=0.005, rely=0.91, relwidth=0.035, relheight=0.08)
main_ax1.set_title('Yield Curve', y=1.05, fontsize=16)
main_ax1.set_xlabel('Maturity (Years)', labelpad=10)
main_ax1.grid(alpha=0.3, animated=True)
main_ax1.set_facecolor('lightblue')
main_ax1.patch.set_alpha(0.3)
rescale_figure_range(main_ax1)


# Window 2 (spots)
fig2, main_ax2 = plt.subplots()
main_ax2.set_position((0.12, 0.22, 0.8, 0.6))
fig2.subplots_adjust(bottom=0.2, top=0.8)
canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
canvas2.get_tk_widget().place(relx=0.005, rely=0.01, relwidth=0.99, relheight=0.98)
refresh2 = ttk.Label(master=frame2,
                     text=interest_process_US.return_refresh_date().strftime('Last updated \n' + '%Y-%m-%d %H:%M:%S'),
                     foreground='white', background='black', justify='center')
refresh2.place(relx=0.815, rely=0.87, relwidth=0.18, relheight=0.12)
interest_data = interest_process_US.return_us_data()
yield_spline = interest_process_US.get_yield_spline()
interest_x_range = np.arange(min(interest_data[0]), max(interest_data[0]) + 1, 0.1)
# Y + T*dY/dT = R(T)
d = yield_spline.derivative()
estimated_rate_process = yield_spline(interest_x_range) + interest_x_range * d(interest_x_range)
interest_lines = [main_ax2.plot(interest_data[0], interest_data[1], c='red', label='Yield')[0],
                  main_ax2.plot(interest_x_range, yield_spline(interest_x_range), label='Cubic Yield Spline')[0],
                  main_ax2.plot(interest_x_range, estimated_rate_process, label='Implied Instantaneous Rates')[0]]
refresh_button2 = ttk.Button(master=frame2, image=refresh_icon, style='WhiteButton.TLabel',
                             command=lambda: refresh_interest(interest_lines))
refresh_button2.place(relx=0.005, rely=0.91, relwidth=0.035, relheight=0.08)
main_ax2.set_title('Implied Instantaneous Rates', y=1.05, fontsize=16)
main_ax2.set_xlabel('Duration (Years)', labelpad=10)
main_ax2.legend()
main_ax2.grid(alpha=0.3, animated=True)
main_ax2.set_facecolor('lightblue')
main_ax2.patch.set_alpha(0.3)


# Window 3 (options)
options_data = OptionsData('^SPX', load=True)
options_df = options_data.return_options_data()
options_expirations, options_expiration_years = options_data.return_options_expirations()
spot = options_data.return_spot()
forward = options_data.return_forward()
smoother = SmoothPrices(options_df, spot)
smoothed_dfs, non_converging, total = return_smoothed_dfs(smoother, N=100)
print(str(non_converging) + ' non-converging solutions out of ' + str(total) + ' data points')

# Price and volatility lines
fig4 = plt.Figure()
main_ax4 = fig4.add_subplot(projection='3d')
canvas4 = FigureCanvasTkAgg(fig4, master=frame3)
canvas4.get_tk_widget().place(relx=0.4, rely=0.01, relwidth=0.598, relheight=0.98)
fig3, main_ax3 = plt.subplots()
main_ax3.set_position((0.12, 0.22, 0.8, 0.6))
canvas3 = FigureCanvasTkAgg(fig3, master=frame3)
canvas3.get_tk_widget().place(relx=0.002, rely=0.01, relwidth=0.498, relheight=0.98)

if len(options_expirations) > 0:
    price_lines = [main_ax3.scatter(options_df[i]['strike'], options_df[i]['CallPrice'], c='yellow',
                                    visible=bool(i == 0), s=4, marker='x') for i in range(len(options_expirations))]
    smoothed_price_lines = [main_ax3.plot(smoothed_dfs[i]['strike'], smoothed_dfs[i]['smoothed'], c='red',
                                          linewidth=1.5, visible=bool(i == 0))[0]
                            for i in range(len(options_expirations))]
    forward_lines = [main_ax3.axvline(x=forward[i], linestyle='--', linewidth=0.5,
                                      visible=bool(i == 0)) for i, T in enumerate(options_expiration_years)]

    yellow_patch = mpl_patches.Patch(color='yellow', label='Market')
    red_patch = mpl_patches.Patch(color='red', label='Smoothed')
    white_patch = mpl_patches.Patch(color='white', label='Forward Price')
    main_ax3.legend(handles=[yellow_patch, red_patch, white_patch])
    lines = [price_lines, smoothed_price_lines, forward_lines]

    rescale_figure_range(main_ax3, False)
    menu = ttk.Combobox(master=frame3, width=15, values=options_expirations)
    menu.set(options_expirations[0])
    menu.bind("<<ComboboxSelected>>", lambda x: change_maturity(lines))
    menu.place(relx=0.38, rely=0.09)
current_index = 0
main_ax3.set_title('SPX Options Prices', y=1.05, fontsize=16)

# Volatility surface
x = np.concatenate([smoothed_dfs[i]['logMoneyness'] for i in range(len(smoothed_dfs))])
y = np.concatenate([np.repeat(options_expiration_years[i],
                              len(smoothed_dfs[i].index)) for i in range(len(smoothed_dfs))])
z = np.concatenate([smoothed_dfs[i]['SmoothVolImp'] for i in range(len(smoothed_dfs))])
points = np.array([x, y]).T
x_grid, y_grid = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
z_grid = interpolate.griddata(points, z, (x_grid, y_grid), method='linear')

surf = main_ax4.plot_surface(x_grid, y_grid, z_grid, rstride=1, cstride=1, cmap=cm.hsv)
main_ax4.view_init(20, 230)
main_ax4.set_title('Volatility Surface', y=1.1, fontsize=16)
main_ax4.set_ylabel('Expiration', labelpad=10)
main_ax4.set_xlabel('Log-Moneyness', labelpad=10)
main_ax4.set_zlabel('IV')
refresh_button3 = ttk.Button(master=frame3, image=refresh_icon, style='WhiteButton.TLabel',
                             command=refresh_options)
refresh_button3.place(relx=0.002, rely=0.91, relwidth=0.019, relheight=0.08)
refresh3 = ttk.Label(master=frame3,
                     text=options_data.return_refresh_date().strftime('Last updated \n' + '%Y-%m-%d %H:%M:%S'),
                     foreground='white', background='black', justify='center')
refresh3.place(relx=0.88, rely=0.86, relwidth=0.1, relheight=0.12)

s.configure('BlackButton.TLabel', background='black')
slider_on1 = ttk.PhotoImage(file=r'./png/on_v.png')
slider_on1 = slider_on1.subsample(10, 10)
slider_off1 = ttk.PhotoImage(file=r'./png/off_v.png')
slider_off1 = slider_off1.subsample(10, 10)
price_vol_label1 = ttk.Label(master=frame3, text=u'\u0024',
                             foreground='white', background='black', justify='center', font=('Helvetica', 12))
price_vol_label1.place(relx=0.015, rely=0.35, relwidth=0.015, relheight=0.05)
price_vol_label2 = ttk.Label(master=frame3, text='\u03c3',
                             foreground='white', background='black', justify='center', font=('Helvetica', 12))
price_vol_label2.place(relx=0.015, rely=0.54, relwidth=0.015, relheight=0.05)
price_vol = 1
price_vol_switch_button = ttk.Button(master=frame3, image=slider_off1, style='BlackButton.TLabel',
                                     command=price_vol_switch)
price_vol_switch_button.place(relx=0.011, rely=0.41, relwidth=0.015, relheight=0.12)

slider_on2 = ttk.PhotoImage(file=r'./png/on_h.png')
slider_on2 = slider_on2.subsample(10, 10)
slider_off2 = ttk.PhotoImage(file=r'./png/off_h.png')
slider_off2 = slider_off2.subsample(10, 10)
strike_moneyness_label1 = ttk.Label(master=frame3, text='K',
                                    foreground='white', background='black', justify='center', font=('Helvetica', 12))
strike_moneyness_label1.place(relx=0.235, rely=0.858, relwidth=0.015, relheight=0.05)
strike_moneyness_label2 = ttk.Label(master=frame3, text=u'\u2c95',
                                    foreground='white', background='black', justify='center', font=('Helvetica', 12))
strike_moneyness_label2.place(relx=0.285, rely=0.852, relwidth=0.04, relheight=0.05)
strike_moneyness = 1
strike_moneyness_switch_button = ttk.Button(master=frame3, image=slider_off2, style='BlackButton.TLabel',
                                            command=strike_moneyness_switch)
strike_moneyness_switch_button.place(relx=0.25, rely=0.82, relwidth=0.03, relheight=0.12)

window.mainloop()

