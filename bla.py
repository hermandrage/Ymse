
import csv
from datetime import datetime
# import panda as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

with open('google_stock_data.csv', 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    # header = next(csv_reader)
    data = [] # [row for row in csv_reader]
    for line in csv_reader:
        date = datetime.strptime(line['Date'], "%m/%d/%Y")
        # print(date)
        open_price = float(line['Open'])
        high = float(line['High'])
        low = float(line['Low'])
        close = float(line['Close'])
        volume = int(line['Volume'])
        Aj = float(line['Adj Close'])
        data.append([date ,open_price ,high ,low ,close ,volume ,Aj])

    with open('new_file.csv' ,'w') as new_file:
        csv_writer = csv.writer(new_file, delimiter='\t')
        csv_writer.writerow(['date', 'value'])

        for i in range(len(data ) -1):
            toadys_date = data[i][0]
            # print(toadys_date)
            todays_price = data[i][1]
            yesterdays_price = data[ i +1][1]
            daily_return = (todays_price - yesterdays_price ) /yesterdays_price
            formated_date = toadys_date.strftime('%Y/%m/%d')
            # print(formated_date)
            csv_writer.writerow([formated_date, daily_return])

with open('new_file.csv', 'r') as plot_file:
    plt_reader = csv.DictReader(plot_file, delimiter='\t')
    # header = next(plt_reader)
    x_axis = []
    y_axis = []
    dates = []
    values = []
    for line in plt_reader:
        dates.append(line['date'])
        values.append(float(line['value']))

    x_values = [datetime.strptime(d ,"%Y/%m/%d").date() for d in dates]
    print(values)
    '''
    ax = plt.gca()
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(formatter)
    locator = mdates.DayLocator()
    ax.xaxis.set_major_locator(locator)

    plt.plot(x_values, values)
    plt.title('daily return')
    plt.savefig('values.png', format='PNG')
    plt.show()
    '''
    plt.plot_date(x_values, values, linestyle='solid')
    plt.gcf().autofmt_xdate()
    plt.show()
