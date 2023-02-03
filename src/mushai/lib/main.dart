import 'package:flutter/material.dart';
import 'dart:io';
import 'package:image_picker/image_picker.dart';
import 'package:tflite/tflite.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        primarySwatch: Colors.red,
      ),
      debugShowCheckedModeBanner: false,
      home: const MyHomePage(title: 'MushAI'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  // This widget is the home page of your application. It is stateful, meaning
  // that it has a State object (defined below) that contains fields that affect
  // how it looks.

  // This class is the configuration for the state. It holds the values (in this
  // case the title) provided by the parent (in this case the App widget) and
  // used by the build method of the State. Fields in a Widget subclass are
  // always marked "final".

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  @override
  void initState() {
    super.initState();
    loadModel();
  }

  @override
  void dispose() {
    Tflite.close();
    super.dispose();
  }

  final ImagePicker _picker = ImagePicker();
  File? _file;
  List<dynamic>? _results;

  Future loadModel() async {
    (await Tflite.loadModel(
      model: "assets/model1.tflite",
      labels: "assets/labels.txt",
    ));
  }

  Future imageClassification(XFile image) async {
    var recognitions = await Tflite.runModelOnImage(
      path: image.path,
      numResults: 2,
      threshold: 0.05,
      imageMean: 127.5,
      imageStd: 127.5,
    );

    setState(() {
      _results = recognitions;
    });
  }

  Future<void> getImage(ImageSource source) async {
    XFile? image = await _picker.pickImage(source: source);
    // ignore: unnecessary_null_comparison
    if (image == null) return;

    File? photo = File(image.path);

    imageClassification(image);
    setState(() {
      _file = photo;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
        centerTitle: true,
      ),
      body: Center(
        child: Column(
          children: [
            Container(
              height: 300,
              width: 300,
              color: Colors.black12,
              margin: const EdgeInsets.only(top: 50, bottom: 50),
              child: _file == null
                  ? const Icon(
                      Icons.image,
                      size: 50,
                    )
                  : Image.file(_file!, fit: BoxFit.fill),
            ),
            SizedBox(
              width: 300,
              child: MaterialButton(
                onPressed: () {
                  getImage(ImageSource.gallery);
                },
                color: Colors.red[700],
                textColor: Colors.white,
                child: Center(
                    child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: const [
                    Padding(
                      padding: EdgeInsets.only(right: 5),
                      child: Icon(
                        Icons.image,
                        size: 20,
                      ),
                    ),
                    Text("From Gallery"),
                  ],
                )),
              ),
            ),
            SizedBox(
              width: 300,
              child: MaterialButton(
                onPressed: () {
                  getImage(ImageSource.camera);
                },
                color: Colors.red[700],
                textColor: Colors.white,
                child: Center(
                    child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: const [
                    Padding(
                      padding: EdgeInsets.only(right: 5),
                      child: Icon(
                        Icons.camera,
                        size: 20,
                      ),
                    ),
                    Text("From Camera"),
                  ],
                )),
              ),
            ),
            _results != null ? 
            Padding(
              padding: const EdgeInsets.only(top:40),
              child:Container(
                width: 200,
                height: 40,
                color: Colors.red[700],
                child: Center(
                  child:Text('${_results![0]["label"]} - ${(_results![0]["confidence"]*100).toStringAsFixed(2)}%',
                  style: const TextStyle(fontWeight: FontWeight.bold,
                  color: Colors.white,
                  ),),
                )
              )
            ): Container()
          ],
        ),
      ),
    );
  }
}
