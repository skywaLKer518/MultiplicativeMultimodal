# Set workspace path here.
WORKSPACE='/'

# Set up absolute path for dataset root here.
DATA_ROOT="$WORKSPACE/data/"
LOGS_DIR="$WORKSPACE/logs/"

# Set up absolute path for model save path here.
SAVE_DIR="$WORKSPACE/result"

# Dataset folders.
CIFAR10_DATA_DIR="$DATA_ROOT/cifar-10"
CIFAR100_DATA_DIR="$DATA_ROOT/cifar-100"


# Relative path to main folder.
LOCAL_DATA_DIR='data'
LOCAL_LOGS_DIR='logs'
LOCAL_SAVE_DIR='results'

# Put 'yes' here to download CIFAR datasets.
DOWNLOAD_CIFAR='yes'


if [ ! -d $DATA_ROOT ]; then
  mkdir -p $DATA_ROOT
fi

mkdir -p $LOCAL_DATA_DIR
if [ $DOWNLOAD_CIFAR = 'yes' ]; then
  cd tools
  ./download_cifar.sh $DATA_ROOT
  cd ..
fi
if [ -d $CIFAR10_DATA_DIR ]; then
  ln -s $CIFAR10_DATA_DIR "$LOCAL_DATA_DIR/cifar-10"
fi
if [ -d $CIFAR10_DATA_DIR ]; then
  ln -s $CIFAR100_DATA_DIR "$LOCAL_DATA_DIR/cifar-100"
fi


mkdir -p $LOCAL_LOGS_DIR
if [ ! -z $LOGS_DIR ]; then
  if [ ! -d $LOGS_DIR ]; then
    mkdir -p $LOGS_DIR
  fi
  ln -s $LOGS_DIR "logs/public"
else
  mkdir -p "logs/public"
fi

if [ ! -z $SAVE_DIR ]; then
  if [ ! -d $SAVE_DIR ]; then
    mkdir -p $SAVE_DIR
  fi
  ln -s $SAVE_DIR $LOCAL_SAVE_DIR
else
  mkdir -p $LOCAL_SAVE_DIR
fi
