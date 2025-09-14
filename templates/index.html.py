< !DOCTYPE
html >
< html
lang = "ru" >
< head >
< meta
charset = "UTF-8" >
< meta
name = "viewport"
content = "width=device-width, initial-scale=1.0" >
< title > Прогнозирование
ОСО - Визуальный
интерфейс < / title >
< style >
body
{
    font - family: Arial, sans - serif;
max - width: 1200
px;
margin: 0
auto;
padding: 20
px;
background - color:  # f5f5f5;
}
.container
{
    background: white;
padding: 30
px;
border - radius: 10
px;
box - shadow: 0
2
px
10
px
rgba(0, 0, 0, 0.1);
}
h1
{
    color:  # 2c3e50;
        text - align: center;
margin - bottom: 30
px;
}
.btn
{
    background:  # 3498db;
        color: white;
padding: 12
px
24
px;
border: none;
border - radius: 5
px;
cursor: pointer;
font - size: 16
px;
margin: 10
px;
transition: background
0.3
s;
}
.btn: hover
{
    background:  # 2980b9;
}
.btn: disabled
{
    background:  # 95a5a6;
        cursor: not -allowed;
}
.btn - success
{
    background:  # 27ae60;
}
.btn - success: hover
{
    background:  # 229954;
}
.btn - danger
{
    background:  # e74c3c;
}
.btn - danger: hover
{
    background:  # c0392b;
}
.status
{
    padding: 15px;
margin: 10
px
0;
border - radius: 5
px;
border - left: 4
px
solid  # 3498db;
}
.status.success
{
    border - color:  # 27ae60;
        background:  # d5f4e6;
}
.status.error
{
    border - color:  # e74c3c;
        background:  # fadbd8;
}
.plot - container
{
    margin: 20px 0;
text - align: center;
}
.plot
{
    max - width: 100 %;
height: auto;
border: 1
px
solid  # ddd;
border - radius: 5
px;
}
.controls
{
    text - align: center;
margin: 30
px
0;
}
.step
{
    margin: 20px 0;
padding: 20
px;
border: 1
px
solid  # ddd;
border - radius: 5
px;
}
.step - number
{
    background:  # 3498db;
        color: white;
width: 30
px;
height: 30
px;
border - radius: 50 %;
display: inline - flex;
align - items: center;
justify - content: center;
margin - right: 10
px;
}
< / style >
< / head >
< body >
< div


class ="container" >

< h1 >🧪 Прогнозирование
Озонового
Слоя < / h1 >

< div


class ="controls" >

< button


class ="btn" onclick="loadData()" > 1. Загрузить данные < / button >

< button


class ="btn" onclick="trainModel()" > 2. Обучить модель < / button >

< button


class ="btn btn-success" onclick="makePredictions()" > 3. Сделать прогнозы < / button >

< button


class ="btn btn-danger" onclick="resetApp()" > Сбросить < / button >

< / div >

< div
id = "status" > < / div >
< div
id = "plots" > < / div >

< div


class ="step" >

< h3 > < span


class ="step-number" > 1 < / span > Загрузка данных < / h3 >

< p > Загружает
данные
о
содержании
озона
из
файла
и
подготавливает
их
для
обучения. < / p >
< / div >

< div


class ="step" >

< h3 > < span


class ="step-number" > 2 < / span > Обучение модели < / h3 >

< p > Обучает
нейронную
сеть
на
загруженных
данных.Используется
архитектура
с
тремя
скрытыми
слоями. < / p >
< / div >

< div


class ="step" >

< h3 > < span


class ="step-number" > 3 < / span > Прогнозирование < / h3 >

< p > Выполняет
прогнозы
на
тестовых
данных
и
сохраняет
результаты.Показывает
графики
сравнения. < / p >
< / div >
< / div >

< script >
function
showStatus(message, isSuccess=true)
{
    const
statusDiv = document.getElementById('status');
statusDiv.innerHTML = `
                      < div


class ="status ${isSuccess ? 'success' : 'error'}" >

< strong >${new
Date().toLocaleTimeString()}: < / strong > ${message}
< / div >
`;
}

function
loadData()
{
showStatus('Загрузка данных...', true);
fetch('/api/load_data', {method: 'POST'})
.then(response= > response.json())
.then(data= > {
    showStatus(data.message, data.success);
updateStatus();
})
.catch(error= > {
    showStatus('Ошибка загрузки: ' + error, false);
});
}

function
trainModel()
{
showStatus('Обучение модели...', true);
fetch('/api/train_model', {method: 'POST'})
.then(response= > response.json())
.then(data= > {
    showStatus(data.message, data.success);
updateStatus();
})
.catch(error= > {
    showStatus('Ошибка обучения: ' + error, false);
});
}

function
makePredictions()
{
showStatus('Выполнение прогнозов...', true);
fetch('/api/predict', {method: 'POST'})
.then(response= > response.json())
.then(data= > {
    showStatus(data.message, data.success);
if (data.success & & data.plots)
{
    showPlots(data.plots);
}
updateStatus();
})
.catch(error= > {
    showStatus('Ошибка прогнозирования: ' + error, false);
});
}

function
showPlots(plots)
{
const
plotsDiv = document.getElementById('plots');
plotsDiv.innerHTML = `
< h2 >📊 Результаты < / h2 >
< div


class ="plot-container" >

< h3 > История
обучения < / h3 >
< img
src = "data:image/png;base64,${plots.training}"


class ="plot" alt="График обучения" >

< / div >
< div


class ="plot-container" >

< h3 > Сравнение
прогнозов < / h3 >
< img
src = "data:image/png;base64,${plots.predictions}"


class ="plot" alt="График прогнозов" >

< / div >
`;
}

function
updateStatus()
{
fetch('/api/status')
.then(response= > response.json())
.then(data= > {
    console.log('Статус приложения:', data);
});
}

function
resetApp()
{
if (confirm('Вы уверены, что хотите сбросить приложение?')) {
location.reload();
}
}

// Загружаем
начальный
статус
updateStatus();
< / script >
    < / body >
        < / html >