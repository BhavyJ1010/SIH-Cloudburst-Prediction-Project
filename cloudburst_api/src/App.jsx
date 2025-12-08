import React, { useState, useEffect } from 'react';
import { 
  Search, 
  Wind, 
  Droplets, 
  Sun, 
  Cloud, 
  CloudRain, 
  CloudSnow, 
  CloudLightning, 
  Moon, 
  Sunrise, 
  Sunset, 
  Thermometer, 
  Navigation, 
  Eye, 
  Umbrella, 
  Menu, 
  X, 
  MapPin, 
  Activity, 
  BarChart2, 
  Star 
} from 'lucide-react';

/**
 * MOCK WEATHER DATA GENERATOR
 */
const generateWeatherData = (city) => {
  const conditions = ['Sunny', 'Cloudy', 'Rainy', 'Stormy', 'Snowy', 'Partly Cloudy', 'Clear Night'];
  const currentCondition = conditions[Math.floor(Math.random() * conditions.length)];
  const baseTemp = Math.floor(Math.random() * (35 - 5) + 5); 
  const basePressure = 1012;
  const baseHumidity = 60;
  const baseWind = 12;

  const getIcon = (cond) => {
    switch(cond) {
      case 'Sunny': return <Sun className="w-full h-full text-amber-400" />;
      case 'Partly Cloudy': return <Cloud className="w-full h-full text-amber-200" />;
      case 'Cloudy': return <Cloud className="w-full h-full text-slate-400" />;
      case 'Rainy': return <CloudRain className="w-full h-full text-blue-400" />;
      case 'Stormy': return <CloudLightning className="w-full h-full text-purple-500" />;
      case 'Snowy': return <CloudSnow className="w-full h-full text-sky-200" />;
      case 'Clear Night': return <Moon className="w-full h-full text-indigo-200" />;
      default: return <Sun className="w-full h-full text-amber-400" />;
    }
  };

  const hourly = Array.from({ length: 24 }, (_, i) => {
    const hour = (new Date().getHours() + i) % 24;
    const isDay = hour > 6 && hour < 18;
    const tempVar = Math.floor(Math.random() * 5 - 2);
    
    const humidityVar = isDay ? -10 + Math.random() * 5 : 10 + Math.random() * 5; 
    const windVar = isDay ? Math.random() * 10 : Math.random() * 5;
    const pressureVar = Math.random() * 4 - 2;

    return {
      time: `${hour === 0 ? 12 : hour > 12 ? hour - 12 : hour} ${hour >= 12 ? 'PM' : 'AM'}`,
      temp: baseTemp + tempVar,
      humidity: Math.min(100, Math.max(0, Math.floor(baseHumidity + humidityVar))),
      wind: Math.floor(baseWind + windVar),
      pressure: Math.floor(basePressure + pressureVar),
      icon: isDay ? <Sun size={20} className="text-amber-400" /> : <Moon size={20} className="text-slate-400" />,
      precip: Math.floor(Math.random() * 30)
    };
  });

  const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const today = new Date().getDay();
  const daily = Array.from({ length: 7 }, (_, i) => {
    const dayIndex = (today + i + 1) % 7;
    const cond = conditions[Math.floor(Math.random() * conditions.length)];
    return {
      day: days[dayIndex],
      high: baseTemp + Math.floor(Math.random() * 5),
      low: baseTemp - Math.floor(Math.random() * 5 + 5),
      condition: cond,
      icon: getIcon(cond),
      precip: Math.floor(Math.random() * 60)
    };
  });

  // Generate AQI Data
  const aqiValue = Math.floor(Math.random() * 120) + 20; 
  let aqiCategory, aqiDesc, aqiColorText, aqiColorBg, aqiGradient;
  
  if (aqiValue <= 50) {
    aqiCategory = 'Good';
    aqiDesc = 'Air quality is satisfactory.';
    aqiColorText = 'text-emerald-600';
    aqiColorBg = 'bg-emerald-100';
    aqiGradient = 'from-teal-400 to-emerald-500';
  } else if (aqiValue <= 100) {
    aqiCategory = 'Moderate';
    aqiDesc = 'Air quality is acceptable.';
    aqiColorText = 'text-yellow-600';
    aqiColorBg = 'bg-yellow-100';
    aqiGradient = 'from-yellow-400 to-orange-500';
  } else {
    aqiCategory = 'Unhealthy';
    aqiDesc = 'Sensitive groups should limit exposure.';
    aqiColorText = 'text-orange-600';
    aqiColorBg = 'bg-orange-100';
    aqiGradient = 'from-orange-400 to-red-500';
  }

  return {
    location: city,
    current: {
      temp: baseTemp,
      condition: currentCondition,
      high: baseTemp + 4,
      low: baseTemp - 3,
      realFeel: baseTemp + 2,
      wind: Math.floor(Math.random() * 20 + 5),
      humidity: Math.floor(Math.random() * 40 + 40),
      uvIndex: Math.floor(Math.random() * 10),
      visibility: Math.floor(Math.random() * 10 + 5),
      pressure: 1012 + Math.floor(Math.random() * 20 - 10),
      dewPoint: baseTemp - 5,
      description: `Today allows for ${currentCondition.toLowerCase()} skies. Winds variable.`,
      sunrise: '6:23 AM',
      sunset: '7:45 PM'
    },
    aqi: {
      value: aqiValue,
      category: aqiCategory,
      description: aqiDesc,
      colorText: aqiColorText,
      colorBg: aqiColorBg,
      gradient: aqiGradient,
      pm25: (aqiValue * 0.4).toFixed(1),
      pm10: (aqiValue * 0.6).toFixed(1),
      so2: (aqiValue * 0.1).toFixed(1),
      no2: (aqiValue * 0.2).toFixed(1)
    },
    hourly,
    daily
  };
};

// --- COMPONENTS ---

// Dynamic Background Bridge Component
const BridgeBackground = ({ colorClass }) => (
  <div className={`absolute bottom-0 left-0 right-0 h-32 flex items-end justify-between px-2 opacity-30 pointer-events-none ${colorClass}`}>
     {/* Bridge Arches Pattern */}
     {[...Array(8)].map((_, i) => (
        <div key={i} className="w-[10%] h-[80%] rounded-t-full bg-current mx-1"></div>
     ))}
     {/* Connecting Deck */}
     <div className="absolute bottom-[80%] left-0 right-0 h-4 bg-current"></div>
     <div className="absolute bottom-[84%] left-0 right-0 h-1 bg-current opacity-50 flex justify-between px-1">
        {[...Array(20)].map((_, i) => (
          <div key={i} className="w-[1px] h-3 bg-current"></div>
        ))}
     </div>
  </div>
);

const HourlyGraph = ({ data, dataKey, color, unit, gradientStart, gradientEnd }) => {
  if (!data || data.length === 0) return null;

  const values = data.map(d => d[dataKey]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const padding = range * 0.2; 
  const graphMin = min - padding;
  const graphMax = max + padding;
  const graphRange = graphMax - graphMin;

  const width = 800; 
  const height = 200; 
  const pointSpacing = width / (data.length - 1);

  const points = data.map((d, i) => {
    const x = i * pointSpacing;
    const y = height - ((d[dataKey] - graphMin) / graphRange) * height;
    return `${x},${y}`;
  }).join(' ');

  const areaPath = `${points} ${width},${height} 0,${height}`;

  return (
    <div className="w-full overflow-x-auto scrollbar-hide">
      <div className="min-w-[600px] h-64 p-4 relative">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full overflow-visible">
          <defs>
            <linearGradient id={`grad-${dataKey}`} x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor={gradientStart} stopOpacity="0.4" />
              <stop offset="100%" stopColor={gradientEnd} stopOpacity="0.0" />
            </linearGradient>
          </defs>
          <line x1="0" y1={height} x2={width} y2={height} stroke="#e2e8f0" strokeWidth="1" />
          <line x1="0" y1="0" x2={width} y2="0" stroke="#e2e8f0" strokeWidth="1" strokeDasharray="5,5" />
          <path d={`M ${areaPath} Z`} fill={`url(#grad-${dataKey})`} />
          <polyline 
            points={points} 
            fill="none" 
            stroke={color} 
            strokeWidth="3" 
            strokeLinecap="round" 
            strokeLinejoin="round" 
          />
          {data.map((d, i) => {
             if (i % 3 !== 0 && i !== data.length - 1) return null;
             const x = i * pointSpacing;
             const y = height - ((d[dataKey] - graphMin) / graphRange) * height;
             return (
               <g key={i}>
                 <circle cx={x} cy={y} r="4" fill="white" stroke={color} strokeWidth="2" />
                 <text 
                   x={x} 
                   y={y - 15} 
                   textAnchor="middle" 
                   fill={color} 
                   className="text-xs font-bold"
                   style={{ fontSize: '24px', fontWeight: 'bold' }} 
                 >
                   {d[dataKey]}{unit}
                 </text>
                 <text 
                   x={x} 
                   y={height + 30} 
                   textAnchor="middle" 
                   fill="#94a3b8" 
                   style={{ fontSize: '20px' }}
                 >
                   {d.time}
                 </text>
               </g>
             );
          })}
        </svg>
      </div>
    </div>
  );
};

const WeatherIcon = ({ condition, className = "w-6 h-6" }) => {
  switch (condition.toLowerCase()) {
    case 'sunny': return <Sun className={`${className} text-amber-400`} />;
    case 'partly cloudy': return <Cloud className={`${className} text-amber-200`} />;
    case 'cloudy': return <Cloud className={`${className} text-slate-300`} />;
    case 'rainy': return <CloudRain className={`${className} text-blue-300`} />;
    case 'stormy': return <CloudLightning className={`${className} text-purple-300`} />;
    case 'snowy': return <CloudSnow className={`${className} text-white`} />;
    case 'clear night': return <Moon className={`${className} text-indigo-200`} />;
    default: return <Sun className={`${className} text-amber-400`} />;
  }
};

const MetricCard = ({ icon, title, value, subtitle }) => (
  <div className="bg-white rounded-3xl p-6 flex flex-col justify-between shadow-lg shadow-slate-200/50 border border-slate-100 transition-all hover:shadow-xl hover:-translate-y-1">
    <div className="flex items-center gap-3 text-teal-600 mb-3">
      <div className="p-2 bg-teal-50 rounded-full">
        {icon}
      </div>
      <span className="text-xs font-bold uppercase tracking-wider text-slate-400">{title}</span>
    </div>
    <div>
      <div className="text-2xl font-bold text-slate-800">{value}</div>
      {subtitle && <div className="text-sm text-slate-400 mt-1 font-medium">{subtitle}</div>}
    </div>
  </div>
);

const HourlyCard = ({ data }) => (
  <div className="flex flex-col items-center justify-between min-w-[80px] p-4 rounded-3xl bg-white hover:bg-teal-50 transition-all border border-slate-100 shadow-md shadow-slate-200/40 group cursor-pointer">
    <span className="text-xs text-slate-400 font-semibold">{data.time}</span>
    <div className="my-3 transition-transform group-hover:scale-110 duration-300">{data.icon}</div>
    <span className="text-xl font-bold text-slate-700">{data.temp}°</span>
    <div className="flex items-center gap-1 mt-2">
      <Droplets size={10} className="text-teal-400" />
      <span className="text-xs text-teal-500 font-medium">{data.precip}%</span>
    </div>
  </div>
);

const DailyRow = ({ data }) => (
  <div className="flex items-center justify-between p-4 hover:bg-teal-50/50 rounded-2xl transition-all cursor-pointer group">
    <span className="text-slate-600 font-semibold w-20">{data.day}</span>
    <div className="flex items-center gap-4 w-32">
      <div className="w-8 h-8">{data.icon}</div>
      <div className="flex items-center gap-1">
        <Droplets size={14} className="text-teal-400 opacity-0 group-hover:opacity-100 transition-opacity" />
        <span className="text-sm text-teal-500 opacity-0 group-hover:opacity-100 transition-opacity font-medium">{data.precip}%</span>
      </div>
    </div>
    <div className="flex items-center gap-6 w-32 justify-end">
      <span className="text-xl font-bold text-slate-800">{data.high}°</span>
      <span className="text-lg font-medium text-slate-400">{data.low}°</span>
    </div>
  </div>
);

const Pollutant = ({ label, value }) => (
  <div className="flex flex-col items-center p-2 bg-slate-50 rounded-xl">
    <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">{label}</span>
    <span className="text-sm font-bold text-slate-600">{value}</span>
  </div>
);

export default function App() {
  const [query, setQuery] = useState('');
  const [weather, setWeather] = useState(null);
  const [loading, setLoading] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [activeHourlyTab, setActiveHourlyTab] = useState('temp'); 

  useEffect(() => {
    handleSearch('New York');
  }, []);

  const handleSearch = (searchTerm) => {
    if (!searchTerm) return;
    setLoading(true);
    setTimeout(() => {
      setWeather(generateWeatherData(searchTerm));
      setLoading(false);
    }, 800);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    handleSearch(query);
  };
  
  // Define tabs data
  const tabs = [
    { id: 'temp', label: 'Temperature' },
    { id: 'wind', label: 'Wind' },
    { id: 'humidity', label: 'Humidity' },
    { id: 'pressure', label: 'Pressure' }
  ];

  // --- Dynamic Theme Logic for Current Weather Card ---
  const getCardTheme = (condition) => {
    const cond = condition.toLowerCase();
    
    // Default (Sunny / Hot style)
    let theme = {
      gradient: "bg-gradient-to-b from-amber-400 to-orange-500",
      bridgeColor: "text-amber-700",
      decor: (
        <div className="absolute inset-0 overflow-hidden rounded-[2.5rem] pointer-events-none">
           <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-64 h-64 bg-amber-300/30 rounded-full blur-2xl"></div>
           <div className="absolute -top-10 -right-10 w-40 h-40 bg-yellow-300 rounded-full blur-3xl opacity-60"></div>
        </div>
      )
    };

    // Rainy / Cloudy (Teal style)
    if (cond.includes('rain') || cond.includes('cloud') || cond.includes('storm')) {
      theme = {
        gradient: "bg-gradient-to-b from-teal-400 to-teal-700",
        bridgeColor: "text-teal-900",
        decor: (
          <div className="absolute inset-0 overflow-hidden rounded-[2.5rem] pointer-events-none">
             {/* Rain drops simulation */}
             {[...Array(20)].map((_, i) => (
                <div 
                  key={i} 
                  className="absolute bg-teal-100/30 w-0.5 rounded-full animate-pulse"
                  style={{
                    height: Math.random() * 20 + 10 + 'px',
                    left: Math.random() * 100 + '%',
                    top: Math.random() * 100 + '%',
                    animationDuration: Math.random() * 1 + 0.5 + 's'
                  }}
                ></div>
             ))}
          </div>
        )
      };
    }

    // Night / Snowy / Clear Night (Dark Blue style)
    if (cond.includes('night') || cond.includes('snow') || cond.includes('clear')) {
      theme = {
        gradient: "bg-gradient-to-b from-indigo-600 to-blue-900",
        bridgeColor: "text-blue-950",
        decor: (
          <div className="absolute inset-0 overflow-hidden rounded-[2.5rem] pointer-events-none">
             {/* Stars */}
             {[...Array(15)].map((_, i) => (
                <Star 
                  key={i} 
                  size={Math.random() * 10 + 4} 
                  className="absolute text-white/40 animate-pulse" 
                  fill="currentColor"
                  style={{
                    left: Math.random() * 100 + '%',
                    top: Math.random() * 60 + '%', // Keep stars in upper part
                    animationDelay: Math.random() * 2 + 's'
                  }}
                />
             ))}
             {/* Moon Glow */}
             <div className="absolute top-10 right-10 w-24 h-24 bg-indigo-300/20 rounded-full blur-xl"></div>
          </div>
        )
      };
    }

    return theme;
  };

  if (!weather && !loading) return <div className="min-h-screen bg-slate-50 flex items-center justify-center text-teal-600 font-medium tracking-wide">Loading SkyCast...</div>;

  return (
    <div className="min-h-screen bg-[#F6F8FC] text-slate-800 font-sans selection:bg-teal-200/50">
      
      {/* Navigation Bar */}
      <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-xl border-b border-slate-100 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-gradient-to-br from-teal-400 to-emerald-500 shadow-lg shadow-teal-500/20">
              <Sun size={24} className="text-white" />
            </div>
            <span className="text-xl font-bold tracking-tight text-slate-800 hidden sm:block">SkyCast</span>
          </div>

          <form onSubmit={handleSubmit} className="flex-1 max-w-xl mx-4 sm:mx-8 relative group">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 w-5 h-5 group-focus-within:text-teal-500 transition-colors" />
            <input
              type="text"
              placeholder="Search city..."
              className="w-full bg-slate-100 border-none rounded-2xl py-3 pl-12 pr-6 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:bg-white transition-all placeholder:text-slate-400 text-slate-700 shadow-inner"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </form>

          <button 
            onClick={() => setMenuOpen(!menuOpen)}
            className="p-2 hover:bg-slate-100 rounded-full transition-colors sm:hidden text-slate-600"
          >
            {menuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
          
          <div className="hidden sm:flex items-center gap-8 text-sm font-semibold text-slate-500">
            <button className="hover:text-teal-600 transition-colors">Maps</button>
            <button className="hover:text-teal-600 transition-colors">News</button>
            <div className="w-10 h-10 rounded-full bg-slate-200 overflow-hidden border-2 border-white shadow-sm">
               <img src="https://api.placeholder.com/150/150" alt="Profile" className="w-full h-full object-cover" />
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-10 pb-24">
        {loading ? (
          <div className="flex flex-col items-center justify-center min-h-[60vh] animate-pulse">
            <Sun size={64} className="text-teal-500 animate-spin-slow mb-6" />
            <div className="text-slate-400 font-medium text-xl tracking-wide">Forecasting...</div>
          </div>
        ) : weather ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            {/* Left Column: Main Weather Info */}
            <div className="lg:col-span-2 space-y-8">
              
              {/* --- DYNAMIC THEME CARD START --- */}
              {(() => {
                const theme = getCardTheme(weather.current.condition);
                return (
                  <div className={`${theme.gradient} rounded-[2.5rem] p-8 sm:p-12 shadow-2xl shadow-gray-400/20 relative overflow-hidden text-white group transition-transform hover:scale-[1.005] duration-500`}>
                    
                    {/* Background Decoration (Sun/Rain/Stars) */}
                    {theme.decor}

                    {/* Bridge Silhouette at Bottom */}
                    <BridgeBackground colorClass={theme.bridgeColor} />

                    <div className="flex flex-col justify-between relative z-10 h-full">
                      <div className="flex justify-between items-start">
                        <div>
                          <div className="flex items-center gap-2 text-white/90 mb-2">
                            <MapPin size={20} className="text-white" />
                            <h2 className="text-3xl font-bold tracking-tight">{weather.location}</h2>
                          </div>
                          <div className="text-white/80 font-medium">Updated {new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</div>
                        </div>
                        <div className="px-4 py-2 bg-white/20 rounded-full text-sm font-bold uppercase tracking-wider backdrop-blur-md shadow-sm border border-white/10">
                          {weather.current.condition}
                        </div>
                      </div>

                      <div className="mt-16 flex flex-col sm:flex-row items-center sm:items-end justify-between gap-12">
                        <div className="flex items-center">
                          <span className="text-8xl sm:text-9xl font-bold tracking-tighter leading-none text-white drop-shadow-sm">{weather.current.temp}°</span>
                          <div className="ml-6 hidden sm:block opacity-90">
                             <WeatherIcon condition={weather.current.condition} className="w-24 h-24 text-white drop-shadow-md" />
                          </div>
                        </div>
                        <div className="space-y-2 text-right w-full sm:w-auto">
                          <div className="text-xl font-medium text-white/90">RealFeel® {weather.current.realFeel}°</div>
                          <div className="text-white/80 font-medium">H: {weather.current.high}° • L: {weather.current.low}°</div>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })()}
              {/* --- DYNAMIC THEME CARD END --- */}

              {/* Hourly Forecast Section */}
              <div className="bg-white rounded-3xl p-6 shadow-lg shadow-slate-200/50 border border-slate-100">
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-6 gap-4">
                  <h3 className="text-xl font-bold text-slate-800 px-2">Hourly Forecast</h3>
                  
                  {/* Tabs */}
                  <div className="flex bg-slate-100 p-1 rounded-xl">
                    {tabs.map(tab => (
                      <button
                        key={tab.id}
                        onClick={() => setActiveHourlyTab(tab.id)}
                        className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition-all ${
                          activeHourlyTab === tab.id 
                            ? 'bg-white text-teal-600 shadow-sm' 
                            : 'text-slate-500 hover:text-slate-700'
                        }`}
                      >
                        {tab.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Tab Content */}
                <div className="transition-all duration-300">
                  {activeHourlyTab === 'temp' && (
                    <div className="flex gap-4 overflow-x-auto pb-6 scrollbar-hide snap-x px-2">
                      {weather.hourly.map((h, i) => (
                        <HourlyCard key={i} data={h} />
                      ))}
                    </div>
                  )}

                  {activeHourlyTab === 'wind' && (
                    <HourlyGraph 
                      data={weather.hourly} 
                      dataKey="wind" 
                      color="#0ea5e9" // Sky blue
                      unit=" km/h"
                      gradientStart="#0ea5e9"
                      gradientEnd="#f0f9ff"
                    />
                  )}

                  {activeHourlyTab === 'humidity' && (
                    <HourlyGraph 
                      data={weather.hourly} 
                      dataKey="humidity" 
                      color="#8b5cf6" // Violet
                      unit="%"
                      gradientStart="#8b5cf6"
                      gradientEnd="#f5f3ff"
                    />
                  )}

                  {activeHourlyTab === 'pressure' && (
                    <HourlyGraph 
                      data={weather.hourly} 
                      dataKey="pressure" 
                      color="#f59e0b" // Amber
                      unit=" mb"
                      gradientStart="#f59e0b"
                      gradientEnd="#fffbeb"
                    />
                  )}
                </div>
              </div>

              {/* Grid Details */}
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-6">
                <MetricCard 
                  icon={<Wind size={20} />} 
                  title="Wind" 
                  value={`${weather.current.wind} km/h`}
                  subtitle="Direction: NE"
                />
                <MetricCard 
                  icon={<Droplets size={20} />} 
                  title="Humidity" 
                  value={`${weather.current.humidity}%`}
                  subtitle={`Dew Point: ${weather.current.dewPoint}°`}
                />
                <MetricCard 
                  icon={<Sun size={20} />} 
                  title="UV Index" 
                  value={weather.current.uvIndex}
                  subtitle={weather.current.uvIndex > 5 ? "High" : "Moderate"}
                />
                <MetricCard 
                  icon={<Eye size={20} />} 
                  title="Visibility" 
                  value={`${weather.current.visibility} km`}
                  subtitle="Clear View"
                />
                <MetricCard 
                  icon={<Navigation size={20} />} 
                  title="Pressure" 
                  value={`${weather.current.pressure} mb`}
                  subtitle="Rising"
                />
                <MetricCard 
                  icon={<Sunrise size={20} />} 
                  title="Sunrise" 
                  value={weather.current.sunrise}
                  subtitle={`Sunset: ${weather.current.sunset}`}
                />
              </div>

            </div>

            {/* Right Column: 7 Day Forecast & Maps */}
            <div className="space-y-8">
              
              {/* 7 Day Forecast */}
              <div className="bg-white rounded-[2.5rem] p-8 shadow-xl shadow-slate-200/60 border border-slate-100">
                <div className="flex items-center justify-between mb-8 px-2">
                  <h3 className="text-xl font-bold text-slate-800">7-Day Forecast</h3>
                  <CloudRain size={20} className="text-teal-500" />
                </div>
                <div className="space-y-2">
                  {weather.daily.map((day, i) => (
                    <DailyRow key={i} data={day} />
                  ))}
                </div>
                <button className="w-full mt-8 py-4 bg-slate-50 hover:bg-slate-100 rounded-2xl text-sm font-bold text-slate-600 transition-all">
                  View 15-Day Outlook
                </button>
              </div>

              {/* Enhanced Air Quality Card */}
              <div className="bg-white rounded-[2.5rem] p-8 shadow-xl shadow-slate-200/60 border border-slate-100 relative overflow-hidden group">
                <div className={`absolute top-0 right-0 w-32 h-32 ${weather.aqi.colorBg} rounded-bl-full -mr-8 -mt-8 z-0 transition-colors duration-500`}></div>
                <div className="relative z-10">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <Activity size={20} className="text-slate-400" />
                      <h3 className="font-bold text-slate-700">Air Quality</h3>
                    </div>
                    <span className={`${weather.aqi.colorBg} ${weather.aqi.colorText} px-3 py-1 rounded-full text-xs font-bold tracking-wide transition-colors duration-500`}>
                      {weather.aqi.category.toUpperCase()}
                    </span>
                  </div>
                  
                  <div className="flex items-baseline gap-2 mb-2">
                    <div className="text-4xl font-bold text-slate-800">{weather.aqi.value}</div>
                    <div className="text-xs font-medium text-slate-400">AQI (US)</div>
                  </div>
                  
                  <div className="text-sm text-slate-500 font-medium mb-6">{weather.aqi.description}</div>
                  
                  {/* AQI Bar */}
                  <div className="h-3 w-full bg-slate-100 rounded-full overflow-hidden mb-6">
                    <div 
                      className={`h-full bg-gradient-to-r ${weather.aqi.gradient} rounded-full transition-all duration-1000`} 
                      style={{ width: `${Math.min((weather.aqi.value / 300) * 100, 100)}%` }}
                    ></div>
                  </div>

                  {/* Pollutants Grid */}
                  <div className="grid grid-cols-4 gap-2">
                    <Pollutant label="PM2.5" value={weather.aqi.pm25} />
                    <Pollutant label="PM10" value={weather.aqi.pm10} />
                    <Pollutant label="SO2" value={weather.aqi.so2} />
                    <Pollutant label="NO2" value={weather.aqi.no2} />
                  </div>
                </div>
              </div>

              {/* Radar Teaser */}
              <div className="bg-white rounded-[2.5rem] p-2 shadow-xl shadow-slate-200/60 border border-slate-100 relative overflow-hidden group cursor-pointer h-48">
                <div className="w-full h-full rounded-[2rem] overflow-hidden relative">
                   <div className="absolute inset-0 bg-[url('https://api.placeholder.com/400/300')] bg-cover bg-center transition-transform duration-700 group-hover:scale-110"></div>
                   <div className="absolute inset-0 bg-teal-900/30 group-hover:bg-teal-900/20 transition-colors"></div>
                   
                   <div className="absolute bottom-0 left-0 w-full p-6 bg-gradient-to-t from-black/60 to-transparent">
                      <div className="flex items-center gap-2 text-white">
                        <Umbrella size={20} />
                        <h3 className="font-bold text-lg">Weather Radar</h3>
                      </div>
                   </div>
                </div>
              </div>

            </div>
          </div>
        ) : null}
      </main>

      {/* Footer */}
      <footer className="mt-16 bg-white border-t border-slate-100">
        <div className="max-w-7xl mx-auto px-6 py-10 flex flex-col md:flex-row items-center justify-between gap-6 text-sm text-slate-400 font-medium">
          <p>© 2024 SkyCast Weather. Mock Data for Demonstration.</p>
          <div className="flex gap-6">
            <a href="#" className="hover:text-teal-600 transition-colors">Privacy</a>
            <a href="#" className="hover:text-teal-600 transition-colors">Terms</a>
            <a href="#" className="hover:text-teal-600 transition-colors">Cookies</a>
          </div>
        </div>
      </footer>
    </div>
  );
}