import xml.etree.ElementTree as et
import csv
tree = et.parse("weather data xml\\data_aqi_cpcb (14).xml")
root = tree.getroot()

arr = []

pm2_array = []
pm10_array = []
no2_array =[]
nh3_array = []
so2_array = []
co_array = []
o3_array = []
aqi_val_array = []
predominant_para_array = []
date_array = []
time_array = []
state_array = []
city_array = []
station_array = []


for country in root.findall("Country"):
    for state in country.findall("State"):
        for city in state.findall("City"):
            for station in city.findall("Station"):
                state_array.append(state.get('id'))
                city_array.append(city.get('id'))
                station_array.append(station.get('id'))
                dt = station.get("lastupdate")
                dt = dt.split()
                date_array.append(dt[0])
                time_array.append(dt[1])
                pol_arr = ["PM2.5","PM10","NO2","NH3","SO2","CO","OZONE"]
                aqi_present = "yes"
                for pindex in station.findall("Pollutant_Index"):
                    #print(pindex.get('Avg'))
                    if(pindex.get('id') == "PM2.5"):
                        pm2_array.append(pindex.get('Avg'))
                        pol_arr.remove('PM2.5')
                    elif(pindex.get('id') == "PM10"):
                        pm10_array.append(pindex.get('Avg'))
                        pol_arr.remove('PM10')
                    elif(pindex.get('id') == 'NO2'):
                        no2_array.append(pindex.get('Avg'))
                        pol_arr.remove("NO2")
                    elif(pindex.get('id') == "NH3"):
                        nh3_array.append(pindex.get('Avg'))
                        pol_arr.remove("NH3")
                    elif(pindex.get('id') == "SO2"):
                        so2_array.append(pindex.get('Avg'))
                        pol_arr.remove("SO2")
                    elif(pindex.get('id') == "CO"):
                        co_array.append(pindex.get('Avg'))
                        pol_arr.remove("CO")
                    elif(pindex.get('id') == "OZONE"):
                        o3_array.append(pindex.get('Avg'))
                        pol_arr.remove("OZONE")
                    
                if(len(pol_arr)!=0):
                    while(len(pol_arr) != 0):
                        ele = pol_arr.pop(0)
                        if(ele == "PM2.5"):
                            pm2_array.append('NA')
                        elif(ele == "PM10"):
                            pm10_array.append('NA')
                        elif(ele == "NO2"):
                            no2_array.append('NA')
                        elif(ele == "NH3"):
                            nh3_array.append('NA')
                        elif(ele == "SO2"):
                            so2_array.append('NA')
                        elif(ele == "CO"):
                            co_array.append('NA')
                        elif(ele == "OZONE"):
                            o3_array.append('NA')
                if(station.find('Air_Quality_Index') is not None):
                    aqi_val_array.append(station.find('Air_Quality_Index').get('Value'))
                    predominant_para_array.append(station.find('Air_Quality_Index').get('Predominant_Parameter'))
                else:
                    aqi_val_array.append('NA')
                    predominant_para_array.append('NA')

data_row = []
for a,b,c,d,e,i,j,k,l,m,n,p,q,r in zip(state_array,city_array,station_array,date_array,time_array,pm2_array,pm10_array,no2_array,nh3_array,so2_array,co_array,o3_array,aqi_val_array,predominant_para_array):
    data_row.append([a,b,c,d,e,i,j,k,l,m,n,p,q,r])

#data_row.insert(0,["state","city","station","date","time","PM2.5","PM10","NO2","NH3","SO2","CO","OZONE","AQI","Predominant_Parameter"])
for i in data_row:
    print(i)    
print("********")


# with open("state_weather_aqi_data_two.csv",'a',newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(data_row)

# print("file written")
