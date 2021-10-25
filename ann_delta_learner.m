% Delta Learning Algorithm with single neuron/layer
clear;close all;clc;

% Parameters:
screen=[-10 10 -10 10];
error_limit=0.001;
counter_limit=4000;
LearningRate=0.05;
w0=0;               % Initial weight vector, 0 for random
PixelDensity=2.5;   % For visualizing the classes of each pixels

figure
title("Mavi sınıf için seç, ENTER")
hold on
axis(screen)
[xi,yi] = getpts;
X=[xi yi];
Y=ones(size(xi,1),1);
plot(xi,yi,'b*')
title("Kırmızı sınıf için seç, ENTER")
[xi,yi] = getpts;
X=[X; xi yi];
Y=[Y ; zeros(size(xi,1),1)];
plot(xi,yi,'r*')

[X,C,S] = normalize(X);

plot(X((Y==1),1),X((Y==1),2),'bo')
plot(X((Y==0),1),X((Y==0),2),'ro')
[n, m] = size(X);

% Initial weight vector creation
if (w0==0); w = 0.01 * randn(m+1,1); else; w = w0; end

% Finding the w vector with least error
error=1;counter=0;
while ((error>error_limit)&&(counter<counter_limit))
    error=0;counter=counter+1;
    for sample = 1:n
        output=sigmoid(w(1)+X(sample,:)*w(2:end));
        w=w+LearningRate*(Y(sample)-output)*(output)*(1-output)*[1 X(sample,:)]';
        error=error+1/2*(Y(sample)-output)^2;
    end
end

% Calculating the accuracy
Z = sigmoid(w(1) + X * w(2:end));
accuracy=mean(Y == (-min(Y)*(Z>0.5)+(Z>0.5)+1*min(Y)));
disp("Tek nöron ve Delta Öğrenme Algoritması için sonuçlar:");
disp(n+" örnek noktasının sınıflandırılmasında %"+100*accuracy+" başarı edildi!")
disp("Eğitim için "+counter+" döngü kullanıldı ve hata "+error+" değerine kadar düşürüldü.")

% Visualition
draw=zeros(((screen(2)-screen(1))/(1/PixelDensity)+1)*((screen(4)-screen(3))/(1/PixelDensity)+1),3);counter=0;
for y=screen(1):1/PixelDensity: screen(2)
    for x=screen(3):1/PixelDensity:screen(4)

        normalized_data=normalize([x y],'center',C,'scale',S);
        xn=normalized_data(1); yn=normalized_data(2);

        net=transpose(w)*[1;xn;yn];
        output=sigmoid(net);
        counter=counter+1;
        if output>0.5
            draw(counter,:)=[x y 1];
        else
            draw(counter,:)=[x y 0];
        end
    end
end
draw_class_1=draw(draw(:,3)==1,:);
draw_class_2=draw(draw(:,3)==0,:);
k=boundary(draw_class_1(:,1),draw_class_1(:,2));
n=boundary(draw_class_2(:,1),draw_class_2(:,2));
title("%"+100*accuracy+ " başarı!")
fill(draw_class_1(k,1),draw_class_1(k,2),'blue','FaceAlpha',0.3,'LineStyle','none');
fill(draw_class_2(n,1),draw_class_2(n,2),'red','FaceAlpha',0.3,'LineStyle','none');
% clearvars -except accuracy error w X Y Z

% figure
% plot(sigmoid(w'*[ones(size(X,1),1) X(:,1) X(:,2)]'))
function Output = sigmoid(Input)      % Y is the array of desired values
Output = 1 ./ (1 + exp(-Input));    % binary activation function
end
