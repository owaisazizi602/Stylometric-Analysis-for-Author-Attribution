#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk


# In[2]:


from nltk import pos_tag, word_tokenize, sent_tokenize
from collections import Counter


# In[3]:


nltk.download('punkt')


# In[4]:


nltk.download('averaged_perceptron_tagger')


# In[12]:


import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize
from collections import Counter
import statistics

# Ensure that NLTK resources are downloaded (if not already)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


actual_author_texts = [
'''
'Twas the night before Christmas, when all thro' the house,
Not a creature was stirring, not even a mouse;
The stockings were hung by the chimney with care,
In hopes that St. Nicholas soon would be there;
The children were nestled all snug in their beds,
While visions of sugar plums danc'd in their heads,
And Mama in her 'kerchief, and I in my cap,
Had just settled our brains for a long winter's nap -
When out on the lawn there arose such a clatter,
I sprung from the bed to see what was the matter.
Away to the window I flew like a flash,
Tore open the shutters, and threw up the sash.
The moon on the breast of the new fallen snow,
Gave the lustre of mid-day to objects below;
When, what to my wondering eyes should appear,
But a minature sleigh, and eight tiny rein-deer,
With a little old driver, so lively and quick,
I knew in a moment it must be St. Nick.
More rapid than eagles his coursers they came,
And he whistled, and shouted, and call'd them by name:
"Now! Dasher, now! Dancer, now! Prancer, and Vixen,
"On! Comet, on! Cupid, on! Dunder and Blixem;
"To the top of the porch! to the top of the wall!
"Now dash away! dash away! dash away all!"
As dry leaves before the wild hurricane fly,
When they meet with an obstacle, mount to the sky;
So up to the house-top the coursers they flew,
With the sleigh full of Toys - and St. Nicholas too:
And then in a twinkling, I heard on the roof
The prancing and pawing of each little hoof.
As I drew in my head, and was turning around,
Down the chimney St. Nicholas came with a bound:
He was dress'd all in fur, from his head to his foot,
And his clothes were all tarnish'd with ashes and soot;
A bundle of toys was flung on his back,
And he look'd like a peddler just opening his pack:
His eyes - how they twinkled! his dimples how merry,
His cheeks were like roses, his nose like a cherry;
His droll little mouth was drawn up like a bow,
And the beard of his chin was as white as the snow;
The stump of a pipe he held tight in his teeth,
And the smoke it encircled his head like a wreath.
He had a broad face, and a little round belly
That shook when he laugh'd, like a bowl full of jelly:
He was chubby and plump, a right jolly old elf,
And I laugh'd when I saw him in spite of myself;
A wink of his eye and a twist of his head
Soon gave me to know I had nothing to dread.
He spoke not a word, but went straight to his work,
And fill'd all the stockings; then turn'd with a jirk,
And laying his finger aside of his nose
And giving a nod, up the chimney he rose.
He sprung to his sleigh, to his team gave a whistle,
And away they all flew, like the down of a thistle:
But I heard him exclaim, ere he drove out of sight -
Happy Christmas to all, and to all a good night.
'''
]

# Disputed texts
disputed_text_1 = '''
Sweet Maid, could wealth or power
Thy heart to love incline,
I would not bless the hour,
The hour that calls thee mine.
Ah! no, beneath the Heaven
Blooms not so fair a flower
As love that's freely given.
Dear youth, have not these eyes,
To thine so oft returning,
Ah! say, have not these tell-tale sighs,
These cheeks with blushes burning,
My every thought bespoken?
Do these denote disguise?
Do these false love betoken?
Oh! bliss, all bliss transcending,
When souls congenial blending,
The sacred flame inspire
Of love's etherial fire.
Such love, from change secure,
For ever shall endure.
True love like this, of heavenly birth,
Not here confin'd to mortal earth,
Shall to immortal Heaven aspire.

'''
disputed_text_2 = '''A vine from noblest lineage sprung
And with the choicest clusters hung,
In purple rob'd, reclining lay,
And catch'd the noontide's fervid ray;
The num'rous plants that deck the field
Did all the palm of beauty yield;
Pronounc'd her fairest of their train
And hail'd her empress of the plain.
A neighb'ring oak whose spiry height
In low-hung clouds was hid from sight,
Who dar'd a thousand howling storms;
Conscious of worth, sublimely stood,
The pride and glory of the wood.

He saw her all defenseless lay
To each invading beast a prey,
And wish'd to clasp her in his arms
And bear her far away from harms.
'Twas love -- 'twas tenderness -- 'twas all
That men the tender passion call.

He urg'd his suit but urg'd in vain,
The vine regardless of his pain
Still flirted with each flippant green
With seeing pleas'd, & being seen;
And as the syren Flattery sang
Would o'er the strains ecstatic hang;
Enjoy'd the minutes as they rose
Nor fears her bosom discompose.

But now the boding clouds arise
And scowling darkness veils the skies;
Harsh thunders roar -- red lightnings gleam,
And rushing torrents close the scene.

The fawning, adulating crowd
Who late in thronged xx bow'd
Now left their goddess of a day
To the O'erwhelming flood a prey,
which swell'd a deluge poured around
& tore her helpless from the ground;
Her rifled foliage floated wide
And ruby nectar ting'd the tide.

With eager eyes and heart dismayed
She look'd but look'd in vain for aid.
"And are my lovers fled," she cry'd,
"Who at my feet this morning sigh'd,
"And swore my reign would never end
"While youth and beauty had a friend?
"I am unhappy who believ'd!
"And they detested who deceived!
"Curse on that whim call'd maiden pride
"Which made me shun the name of bride
"When yonder oak confessed his flame
"And woo'd me in fair honor's name.
"But now repentance comes too late
"And all forlorn, I meet my fate."

The oak who safely wav'd above
Look'd down once more with eyes of love
(Love higher wrought with pity join'd
True mark of an exalted mind,)
Declared her coldness could suspend
But not his gen'rous passion end.
Beg'd to renew his am'rous plea,
As warm for union now as he,
To his embraces quick she flew
And felt & gave sensations new.

Enrich'd & graced by the sweet prise
He lifts her tendrils to the skies;
Whilst she, protected and carest,
Sinks in his arms completely blest.
'''
disputed_text_3 = '''Home of the Percy’s high-born race,
Home of their beautiful and brave,
Alike their birth and burial-place,
Their cradle and their grave!
Still sternly o’er the castle gate
Their house’s Lion stands in state,
As in his proud departed hours;
And warriors frown in stone on high,
And feudal banners “flout the sky”
Above his princely towers.

A gentle hill its side inclines,
Lovely in England’s fadeless green,
To meet the quiet stream which winds
Through this romantic scene
As silently and sweetly still,
As when, at evening, on that hill,
While summer’s wind blew soft and low,
Seated by gallant Hotspur’s side,
His Katherine was a happy bride,
A thousand years ago.

Gaze on the Abbey’s ruined pile:
Does not the succoring ivy, keeping
Her watch around it, seem to smile,
As o’er a loved one sleeping?
One solitary turret gray
Still tells, in melancholy glory,
The legend of the Cheviot day,
The Percy’s proudest border story.
That day its roof was triumph’s arch;
Then rang, from aisle to pictured dome,
The light step of the soldier’s march,
The music of the trump and drum;
And babe, and sire, the old, the young,
And the monk’s hymn, and minstrel’s song,
And woman’s pure kiss, sweet and long,
Welcomed her warrior home.

Wild roses by the Abbey towers
Are gay in their young bud and bloom:
They were born of a race of funeral-flowers
That garlanded, in long-gone hours,
A templar’s knightly tomb.
He died, the sword in his mailed hand,
On the holiest spot of the Blessed land,
Where the Cross was damped with his dying breath,
When blood ran free as festal wine,
And the sainted air of Palestine
Was thick with the darts of death.

Wise with the lore of centuries,
What tales, if there be “tongues in trees,”
Those giant oaks could tell,
Of beings born and buried here;
Tales of the peasant and the peer,
Tales of the bridal and the bier,
The welcome and farewell,
Since on their boughs the startled bird
First, in her twilight slumbers, heard
The Norman’s curfew-bell!

I wandered through the lofty halls
Trod by the Percys of old fame,
And traced upon the chapel walls
Each high, heroic name,
From him3 who once his standard set
Where now, o’er mosque and minaret,
Glitter the Sultan’s crescent moons;
To him who, when a younger son,
Fought for King George at Lexington,4
A major of dragoons.

That last half stanza—it has dashed
From my warm lip the sparkling cup;
The light that o’er my eyebeam flashed,
The power that bore my spirit up
Above this bank-note world—is gone;
And Alnwick’s but a market town,
And this, alas! its market day,

And beasts and borderers throng the way;
Oxen and bleating lambs in lots,
Northumbrian boors and plaided Scots,
Men in the coal and cattle line;
From Teviot’s bard and hero land,
From royal Berwick’s5 beach of sand,
From Wooller, Morpeth, Hexham, and
Newcastle-upon-Tyne.

These are not the romantic times
So beautiful in Spenser’s rhymes,
So dazzling to the dreaming boy:
Ours are the days of fact, not fable,
Of knights, but not of the round table,
Of Bailie Jarvie, not Rob Roy:
’Tis what “our President,” Monroe,
Has called “the era of good feeling:”
The Highlander, the bitterest foe
To modern laws, has felt their blow,
Consented to be taxed, and vote,
And put on pantaloons and coat,
And leave off cattle-stealing:
Lord Stafford mines for coal and salt,
The Duke of Norfolk deals in malt,
The Douglass in red herrings;
And noble name and cultured land,
Palace, and park, and vassal-band,
Are powerless to the notes of hand
Of Rothschild or the Barings.

The age of bargaining, said Burke,
Has come: to-day the turbaned Turk
(Sleep, Richard of the lion heart!
Sleep on, nor from your cerements start)
Is England’s friend and fast ally;
The Moslem tramples on the Greek,
And on the Cross and altar-stone,
And Christendom looks tamely on,
And hears the Christian maiden shriek,
And sees the Christian father die;
And not a sabre-blow is given
For Greece and fame, for faith and heaven,
By Europe’s craven chivalry.

You’ll ask if yet the Percy lives
In the armed pomp of feudal state?
The present representatives
Of Hotspur and his “gentle Kate,”
Are some half-dozen serving-men
In the drab coat of William Penn;
A chambermaid, whose lip and eye,
And cheek, and brown hair, bright and curling,
Spoke Nature’s aristocracy;
And one, half groom, half seneschal,
Who bowed me through court, bower, and hall,
From donjon-keep to turret wall,
For ten-and-sixpence sterling.'''

# Function to analyze text features
def analyze_text(text):
    sentences = sent_tokenize(text)
    avg_len = sum(len(word_tokenize(sent)) for sent in sentences) / len(sentences) if sentences else 0
    
    pos_tags = pos_tag(word_tokenize(text))
    pos_freq = Counter(tag for word, tag in pos_tags)
    
    punctuation = Counter(char for char in text if char in '.,;!?')
    
    return avg_len, pos_freq, punctuation

# Analyze actual author's corpus
author_avg_lengths = []
author_pos_freqs = Counter()
author_punctuations = Counter()

for text in actual_author_texts:
    avg_len, pos_freq, punctuation = analyze_text(text)
    author_avg_lengths.append(avg_len)
    author_pos_freqs.update(pos_freq)
    author_punctuations.update(punctuation)

# Average statistics for the actual author
author_avg_length = statistics.mean(author_avg_lengths) if author_avg_lengths else 0
author_pos_freqs_normalized = {k: v / len(actual_author_texts) for k, v in author_pos_freqs.items()}
author_punctuations_normalized = {k: v / len(actual_author_texts) for k, v in author_punctuations.items()}

# Analyze the first disputed text
disputed_avg_length_1, disputed_pos_freqs_1, disputed_punctuations_1 = analyze_text(disputed_text_1)

# Analyze the second disputed text
disputed_avg_length_2, disputed_pos_freqs_2, disputed_punctuations_2 = analyze_text(disputed_text_2)

# Analyze the third disputed text
disputed_avg_length_3, disputed_pos_freqs_3, disputed_punctuations_3 = analyze_text(disputed_text_3)

# Compare sentence lengths
print(f"Average Sentence Length - Actual Author: {author_avg_length}")
print(f"Disputed Text 1: {disputed_avg_length_1}, Disputed Text 2: {disputed_avg_length_2}, Disputed Text 3: {disputed_avg_length_3}")

# Compare POS frequencies
print(f"POS Frequency - Actual Author: {author_pos_freqs_normalized}")
print(f"Disputed Text 1: {disputed_pos_freqs_1}")
print(f"Disputed Text 2: {disputed_pos_freqs_2}")
print(f"Disputed Text 3: {disputed_pos_freqs_3}")

# Compare punctuation usage
print(f"Punctuation - Actual Author: {author_punctuations_normalized}")
print(f"Punctuation - Disputed Text 1: {disputed_punctuations_1}")
print(f"Punctuation - Disputed Text 2: {disputed_punctuations_2}")
print(f"Punctuation - Disputed Text 3: {disputed_punctuations_3}")


# In[13]:


import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize
from collections import Counter
import statistics

# Ensure that NLTK resources are downloaded (if not already)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


actual_author_texts = [
'''
'Twas the night before Christmas, when all thro' the house,
Not a creature was stirring, not even a mouse;
The stockings were hung by the chimney with care,
In hopes that St. Nicholas soon would be there;
The children were nestled all snug in their beds,
While visions of sugar plums danc'd in their heads,
And Mama in her 'kerchief, and I in my cap,
Had just settled our brains for a long winter's nap -
When out on the lawn there arose such a clatter,
I sprung from the bed to see what was the matter.
Away to the window I flew like a flash,
Tore open the shutters, and threw up the sash.
The moon on the breast of the new fallen snow,
Gave the lustre of mid-day to objects below;
When, what to my wondering eyes should appear,
But a minature sleigh, and eight tiny rein-deer,
With a little old driver, so lively and quick,
I knew in a moment it must be St. Nick.
More rapid than eagles his coursers they came,
And he whistled, and shouted, and call'd them by name:
"Now! Dasher, now! Dancer, now! Prancer, and Vixen,
"On! Comet, on! Cupid, on! Dunder and Blixem;
"To the top of the porch! to the top of the wall!
"Now dash away! dash away! dash away all!"
As dry leaves before the wild hurricane fly,
When they meet with an obstacle, mount to the sky;
So up to the house-top the coursers they flew,
With the sleigh full of Toys - and St. Nicholas too:
And then in a twinkling, I heard on the roof
The prancing and pawing of each little hoof.
As I drew in my head, and was turning around,
Down the chimney St. Nicholas came with a bound:
He was dress'd all in fur, from his head to his foot,
And his clothes were all tarnish'd with ashes and soot;
A bundle of toys was flung on his back,
And he look'd like a peddler just opening his pack:
His eyes - how they twinkled! his dimples how merry,
His cheeks were like roses, his nose like a cherry;
His droll little mouth was drawn up like a bow,
And the beard of his chin was as white as the snow;
The stump of a pipe he held tight in his teeth,
And the smoke it encircled his head like a wreath.
He had a broad face, and a little round belly
That shook when he laugh'd, like a bowl full of jelly:
He was chubby and plump, a right jolly old elf,
And I laugh'd when I saw him in spite of myself;
A wink of his eye and a twist of his head
Soon gave me to know I had nothing to dread.
He spoke not a word, but went straight to his work,
And fill'd all the stockings; then turn'd with a jirk,
And laying his finger aside of his nose
And giving a nod, up the chimney he rose.
He sprung to his sleigh, to his team gave a whistle,
And away they all flew, like the down of a thistle:
But I heard him exclaim, ere he drove out of sight -
Happy Christmas to all, and to all a good night.
'''
]

# Disputed texts
disputed_text_1 = '''
Sweet melancholy Bard! whose piercing thought
Found humblest themes with pure instruction fraught;
How hard for mortal sight to trace the ways
Of Heav'n throughout thy life's mysterious maze!
Why was it order'd that thy gentle mind,
Which fancy fir'd and piety refin'd,
Should in this guilty world be forc'd to dwell,
Like some base culprit in his gloomy cell,
Rous'd from its due repose by feverish dreams,
By goblin forms, by din of fancied screams?
Why was that fertile genius waste and chill'd?
By wintry blasts its opening blossoms kill'd?
A soil where Yemen's spicy buds might blow,
And Persia's rose a purer fragrance know!
Why bloom'd so late those sweet poetic flowers,
Bless'd by no summer suns, no vernal showers,
Which in the autumn of thy days were rear'd
By friendship's dew, by fickle zephyrs cheer'd?
I hear a distant Seraph bid me "Hold,
Nor tempt high Heav'n by such inquiries bold.
Weak-sighted mortal! canst thou not discern.
What from unaided reason thou might'st learn?
Had fortune's sunbeams cheer'd his early days,
Amidst the soft favonian breath of praise,
Those fruitful virtues which sprang up so fair,
Those blossoms breathing odors on the air,
By weeds of pride and vanity o'ergrown,
Unheeded might have bloom'd, and died unknown.
Presumptuous mortal 'twould become thee well
On this thy fellow mortal's life to dwell;
For in his breast, when rack'd by fiercest woes,
To question Heav'n, no daring thought e'er rose.
His actions vice and folly view with shame;
His precepts foul-mouth'd envy dares not blame;
His well-lov'd image still calls many a tear;
His cherish'd name all ages shall revere."

'''
disputed_text_2 = '''
In long gone years a fox and crane
Were bound in friendship's golden chain;
Whene'er they met, the fox would bow
And madame Crane would curtsie low-
-My lovely Crane how do you do?
-I'm very well; pray how are you?
Thus time passed on, both very civil
Till Reynard in an hour evil
Projected what he thought a stroke
The world would call a pretty joke -
A billet wrote on gilded paper
And sealed it with a perfumed wafer
Announced the day, if she saw fit
To take a tete-a-tetetit-bit;
The day arriv'd -she preen'd each feather
And summon'd ev'ry grace together;
At breakfast scarce a morsel eat
Intent to riot at the treat -

She came; wide stood the unfolded door
And roses deck'd the sanded floor -
- There hyacinths in festoons hung
- Here lillies their rich fragrance flung -

The table drawn; the damask laid
And soup prepared of bullock's marrow
Pour'd in each plate profuse; but shallow;
The fox began to lap in haste
And made a plentiful repast,
Pressed his fair friend to do the same
And to encourage, lap'd again -

The Crane be sure with her long beak
Could not a single morsel pick;
She felt the bite--but little said
And very soon her exit made,
Just beg'd the fox would come next day
And sup with her in her plain way;
Reynard declared she did him honor
-He certainly would wait upon her.

Her domicile was well prepar'd
No cost or labor had been spared;
Roses and tulips on the floor
And daffodils the ceiling bore;
Nor was a band of music wanting
For whippoorwills and frogs were chanting.
The sun had set and given way
To sober evening's mantle gray;
The fox arriv'd with stomach keen
-Hoped he saw in health his Queen
And added in his courtliest air
She ne'er before had look'd so fair.

The Crane replied in mildest mood
That all he said was very good,
She meekly meant to do her duty
And ne'er dream'd of praise or beauty.
-She spoke; The table soon was spread
And ev'rything in order paid;
Two narrow jars now graced the board
With nicely minced ven'son stored;
- Now let's fall to, sir, if you will--
And in she pok'd her slender bill
And gulp'd of viands at her leisure
- To see you eat would give me pleasure
She cried; eat, neighbor, eat
I fear you do not like my treat;
It suits my palate to a hair
Pray, Chummy, eat and do not spare.
- The fox looked on with rueful phys
Feeling in all its force the quiz.

The Crane enjoy'd his discontent
And thus address'd him as he went,
The truest adage ever spoke
Was "He that Gives must Take a joke."

H.L. to his beloved daughter Jane, Feb. 19, 1827.


'''
disputed_text_3 = '''BURNS.

TO A ROSE, BROUGHT FROM NEAR ALLOWAY KIRK, IN AYRSHIRE, IN THE AUTUMN OF 1822.

Wild Rose of Alloway! my thanks;
Thou ’mindst me of that autumn noon
When first we met upon “the banks
And braes o’ bonny Doon.”

Like thine, beneath the thorn-tree’s bough,
My sunny hour was glad and brief,
We’ve crossed the winter sea, and thou
Art withered—flower and leaf.

And will not thy death-doom be mine—
The doom of all things wrought of clay—
And withered my life’s leaf like thine,
Wild rose of Alloway?

Not so his memory, for his sake
My bosom bore thee far and long,
His—who a humbler flower could make
Immortal as his song,

The memory of Burns—a name
That calls, when brimmed her festal cup,
A nation’s glory and her shame,
In silent sadness up.

A nation’s glory—be the rest
Forgot—she’s canonized his mind;
And it is joy to speak the best
We may of human kind.

I’ve stood beside the cottage-bed
Where the Bard-peasant first drew breath;
A straw-thatched roof above his head,
A straw-wrought couch beneath.

And I have stood beside the pile,
His monument—that tells to Heaven
The homage of earth’s proudest isle
To that Bard-peasant given!

Bid thy thoughts hover o’er that spot,
Boy-minstrel, in thy dreaming hour;
And know, however low his lot,
A Poet’s pride and power:

The pride that lifted Burns from earth,
The power that gave a child of song
Ascendency o’er rank and birth,
The rich, the brave, the strong;

And if despondency weigh down
Thy spirit’s fluttering pinions then,
Despair—thy name is written on
The roll of common men.

There have been loftier themes than his,
And longer scrolls, and louder lyres,
And lays lit up with Poesy’s
Purer and holier fires:

Yet read the names that know not death;
Few nobler ones than Burns are there;
And few have won a greener wreath
Than that which binds his hair.

His is that language of the heart,
In which the answering heart would speak,
Thought, word, that bids the warm tear start,
Or the smile light the cheek;

And his that music, to whose tone
The common pulse of man keeps time,
In cot or castle’s mirth or moan,
In cold or sunny clime.

And who hath heard his song, nor knelt
Before its spell with willing knee,
And listened, and believed, and felt
The Poet’s mastery:

O’er the mind’s sea, in calm and storm,
O’er the heart’s sunshine and its showers,
O’er Passion’s moments bright and warm,
O’er Reason’s dark, cold hours;

On fields where brave men “die or do,”
In halls where rings the banquet’s mirth,
Where mourners weep, where lovers woo,
From throne to cottage-hearth?

What sweet tears dim the eye unshed,
What wild vows falter on the tongue,
When “Scots wha hae wi’ Wallace bled,”
Or “Auld Lang Syne” is sung!

Pure hopes, that lift the soul above,
Come with his Cotter’s hymn of praise,
And dreams of youth, and truth, and love,
With “Logan’s” banks and braes.

And when he breathes his master-lay
Of Alloway’s witch-haunted wall,
All passions in our frames of clay
Come thronging at his call.

Imagination’s world of air,
And our own world, its gloom and glee,
Wit, pathos, poetry, are there,
And death’s sublimity.

And Burns—though brief the race he ran,
Though rough and dark the path he trod,
Lived—died—in form and soul a Man,
The image of his God.

Through care, and pain, and want, and woe,
With wounds that only death could heal,
Tortures—the poor alone can know,
The proud alone can feel;

He kept his honesty and truth,
His independent tongue and pen,
And moved, in manhood as in youth,
Pride of his fellow-men.

Strong sense, deep feeling, passions strong,
A hate of tyrant and of knave,
A love of right, a scorn of wrong,
Of coward and of slave;

A kind, true heart, a spirit high,
That could not fear and would not bow,
Were written in his manly eye
And on his manly brow.

Praise to the bard! his words are driven,
Like flower-seeds by the far winds sown,
Where’er, beneath the sky of heaven,
The birds of fame have flown.

Praise to the man! a nation stood
Beside his coffin with wet eyes,
Her brave, her beautiful, her good,
As when a loved one dies.

And still, as on his funeral-day,
Men stand his cold earth-couch around,
With the mute homage that we pay
To consecrated ground.

And consecrated ground it is,
The last, the hallowed home of one
Who lives upon all memories,
Though with the buried gone.

Such graves as his are pilgrim-shrines,
Shrines to no code or creed confined—
The Delphian vales, the Palestines,
The Meccas of the mind.

Sages, with wisdom’s garland wreathed,
Crowned kings, and mitred priests of power,
And warriors with their bright swords sheathed,
The mightiest of the hour;

And lowlier names, whose humble home
Is lit by fortune’s dimmer star,
Are there—o’er wave and mountain come,
From countries near and far;

Pilgrims whose wandering feet have pressed
The Switzer’s snow, the Arab’s sand,
Or trod the piled leaves of the West,
My own green forest-land.

All ask the cottage of his birth,
Gaze on the scenes he loved and sung,
And gather feelings not of earth
His fields and streams among.

They linger by the Doon’s low trees,
And pastoral Nith, and wooded Ayr,
And round thy sepulchres, Dumfries!
The poet’s tomb is there.

But what to them the sculptor’s art,
His funeral columns, wreaths and urns?
Wear they not graven on the heart
The name of Robert Burns?'''

# Function to analyze text features
def analyze_text(text):
    sentences = sent_tokenize(text)
    avg_len = sum(len(word_tokenize(sent)) for sent in sentences) / len(sentences) if sentences else 0
    
    pos_tags = pos_tag(word_tokenize(text))
    pos_freq = Counter(tag for word, tag in pos_tags)
    
    punctuation = Counter(char for char in text if char in '.,;!?')
    
    return avg_len, pos_freq, punctuation

# Analyze actual author's corpus
author_avg_lengths = []
author_pos_freqs = Counter()
author_punctuations = Counter()

for text in actual_author_texts:
    avg_len, pos_freq, punctuation = analyze_text(text)
    author_avg_lengths.append(avg_len)
    author_pos_freqs.update(pos_freq)
    author_punctuations.update(punctuation)

# Average statistics for the actual author
author_avg_length = statistics.mean(author_avg_lengths) if author_avg_lengths else 0
author_pos_freqs_normalized = {k: v / len(actual_author_texts) for k, v in author_pos_freqs.items()}
author_punctuations_normalized = {k: v / len(actual_author_texts) for k, v in author_punctuations.items()}

# Analyze the first disputed text
disputed_avg_length_1, disputed_pos_freqs_1, disputed_punctuations_1 = analyze_text(disputed_text_1)

# Analyze the second disputed text
disputed_avg_length_2, disputed_pos_freqs_2, disputed_punctuations_2 = analyze_text(disputed_text_2)

# Analyze the third disputed text
disputed_avg_length_3, disputed_pos_freqs_3, disputed_punctuations_3 = analyze_text(disputed_text_3)

# Compare sentence lengths
print(f"Average Sentence Length - Actual Author: {author_avg_length}")
print(f"Disputed Text 1: {disputed_avg_length_1}, Disputed Text 2: {disputed_avg_length_2}, Disputed Text 3: {disputed_avg_length_3}")

# Compare POS frequencies
print(f"POS Frequency - Actual Author: {author_pos_freqs_normalized}")
print(f"Disputed Text 1: {disputed_pos_freqs_1}")
print(f"Disputed Text 2: {disputed_pos_freqs_2}")
print(f"Disputed Text 3: {disputed_pos_freqs_3}")

# Compare punctuation usage
print(f"Punctuation - Actual Author: {author_punctuations_normalized}")
print(f"Punctuation - Disputed Text 1: {disputed_punctuations_1}")
print(f"Punctuation - Disputed Text 2: {disputed_punctuations_2}")
print(f"Punctuation - Disputed Text 3: {disputed_punctuations_3}")


# In[24]:


import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize
from collections import Counter
import statistics

# Ensure that NLTK resources are downloaded (if not already)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


actual_author_texts = [
'''
'Twas the night before Christmas, when all thro' the house,
Not a creature was stirring, not even a mouse;
The stockings were hung by the chimney with care,
In hopes that St. Nicholas soon would be there;
The children were nestled all snug in their beds,
While visions of sugar plums danc'd in their heads,
And Mama in her 'kerchief, and I in my cap,
Had just settled our brains for a long winter's nap -
When out on the lawn there arose such a clatter,
I sprung from the bed to see what was the matter.
Away to the window I flew like a flash,
Tore open the shutters, and threw up the sash.
The moon on the breast of the new fallen snow,
Gave the lustre of mid-day to objects below;
When, what to my wondering eyes should appear,
But a minature sleigh, and eight tiny rein-deer,
With a little old driver, so lively and quick,
I knew in a moment it must be St. Nick.
More rapid than eagles his coursers they came,
And he whistled, and shouted, and call'd them by name:
"Now! Dasher, now! Dancer, now! Prancer, and Vixen,
"On! Comet, on! Cupid, on! Dunder and Blixem;
"To the top of the porch! to the top of the wall!
"Now dash away! dash away! dash away all!"
As dry leaves before the wild hurricane fly,
When they meet with an obstacle, mount to the sky;
So up to the house-top the coursers they flew,
With the sleigh full of Toys - and St. Nicholas too:
And then in a twinkling, I heard on the roof
The prancing and pawing of each little hoof.
As I drew in my head, and was turning around,
Down the chimney St. Nicholas came with a bound:
He was dress'd all in fur, from his head to his foot,
And his clothes were all tarnish'd with ashes and soot;
A bundle of toys was flung on his back,
And he look'd like a peddler just opening his pack:
His eyes - how they twinkled! his dimples how merry,
His cheeks were like roses, his nose like a cherry;
His droll little mouth was drawn up like a bow,
And the beard of his chin was as white as the snow;
The stump of a pipe he held tight in his teeth,
And the smoke it encircled his head like a wreath.
He had a broad face, and a little round belly
That shook when he laugh'd, like a bowl full of jelly:
He was chubby and plump, a right jolly old elf,
And I laugh'd when I saw him in spite of myself;
A wink of his eye and a twist of his head
Soon gave me to know I had nothing to dread.
He spoke not a word, but went straight to his work,
And fill'd all the stockings; then turn'd with a jirk,
And laying his finger aside of his nose
And giving a nod, up the chimney he rose.
He sprung to his sleigh, to his team gave a whistle,
And away they all flew, like the down of a thistle:
But I heard him exclaim, ere he drove out of sight -
Happy Christmas to all, and to all a good night.
'''
]

# Disputed texts
disputed_text_1 = '''

This name here drawn by Flora's hand
Portrays, alas! her mind:
The beating surf and yielding sand
Soon leave no trace behind.
But Flora's name shall still abide
In many a bosom trac'd,
Not e'en by time's destroying tide
Nor fortune's storms effac'd.


'''
disputed_text_2 = '''
An elegy on the death of MONTGOMERY TAPPEN who dies at 
Poughkeepsie on the 20th of Nov. 1784 in the ninth year of his age.

The sweetest, gentlest, of the youthful train,
Here lies his clay cold upon the sable bier!
He scarce had started on life's varied plain,
For dreary death arrested his career.

His cheek might vie with the expanded rose,
And Genius sparkled in his azure eyes!
A victim so unblemish'd Heaven chose,
And bore the beauteous lambkin to the skies.

Adieu thou loveliest child! Adieu adieu!
Our wishes fain would follow thee on high.
What more can friendship; what more fondness do,
But drop the unbidden tear & heave the sigh?

Ye youths whose ardent bosoms virtue fires,
Who eager wish applause and pant for fame,
Press round MONTGOMERY'S hearse, the NAME inspires
And lights in kindred souls its native flame.

COLUMBIA grateful hails the tender sound
And when MONTGOMERY'S nam'd still drops a tear,
From shore to shore to earth's remotest bound
Where LIBERTY is known that NAME is dear.




'''
disputed_text_3 = '''A Chief of the Indian Tribes, the Tuscaroras.

ON LOOKING AT HIS PORTRAIT BY WEIR.

Cooper, whose name is with his country’s woven,
First in her files, her Pioneer of mind—
A wanderer now in other climes, has proven
His love for the young land he left behind;

And throned her in the senate-hall of nations,
Robed like the deluge rainbow, heaven-wrought;
Magnificent as his own mind’s creations,
And beautiful as its green world of thought:

And faithful to the Act of Congress, quoted
As law authority, it passed nem. con.:
He writes that we are, as ourselves have voted,
The most enlightened people ever known:

That all our week is happy as a Sunday
In Paris, full of song, and dance, and laugh;
And that, from Orleans to the Bay of Fundy,
There’s not a bailiff or an epitaph:

And furthermore—in fifty years, or sooner,
We shall export our poetry and wine;
And our brave fleet, eight frigates and a schooner,
Will sweep the seas from Zembla to the Line.

If he were with me, King of Tuscarora!
Gazing, as I, upon thy portrait now,
In all its medalled, fringed, and beaded glory,
Its eye’s dark beauty, and its thoughtful brow—

Its brow, half martial and half diplomatic,
Its eye, upsoaring like an eagle’s wings;
Well might he boast that we, the Democratic,
Outrival Europe, even in our Kings!

For thou wast monarch born. Tradition’s pages
Tell not the planting of thy parent tree,
But that the forest tribes have bent for ages
To thee, and to thy sires, the subject knee.

Thy name is princely—if no poet’s magic
Could make Red Jacket grace an English rhyme,
Though some one with a genius for the tragic
Hath introduced it in a pantomime—

Yet it is music in the language spoken
Of thine own land, and on her herald-roll;
As bravely fought for, and as proud a token
As Cœur de Lion’s of a warrior’s soul.

Thy garb—though Austria’s bosom-star would frighten
That medal pale, as diamonds the dark mine,
And George the Fourth wore, at his court at Brighton
A more becoming evening dress than thine;

Yet ’tis a brave one, scorning wind and weather,
And fitted for thy couch, on field and flood,
As Rob Roy’s tartan for the Highland heather,
Or forest green for England’s Robin Hood.

Is strength a monarch’s merit, like a whaler’s?
Thou art as tall, as sinewy, and as strong
As earth’s first kings—the Argo’s gallant sailors,
Heroes in history and gods in song.

Is beauty?—Thine has with thy youth departed;
But the love-legends of thy manhood’s years,
And she who perished, young and broken-hearted,
Are—but I rhyme for smiles and not for tears.

Is eloquence?—Her spell is thine that reaches
The heart, and makes the wisest head its sport;
And there’s one rare, strange virtue in thy speeches,
The secret of their mastery—they are short.

The monarch mind, the mystery of commanding,
The birth-hour gift, the art Napoleon,
Of winning, fettering, moulding, wielding, banding
The hearts of millions till they move as one:

Thou hast it. At thy bidding men have crowded
The road to death as to a festival;
And minstrels, at their sepulchres, have shrouded
With banner-folds of glory the dark pall.

Who will believe? Not I—for in deceiving
Lies the dear charm of life’s delightful dream;
I cannot spare the luxury of believing
That all things beautiful are what they seem;

Who will believe that, with a smile whose blessing
Would, like the Patriarch’s, soothe a dying hour,
With voice as low, as gentle, and caressing,
As e’er won maiden’s lip in moonlit bower:

With look like patient Job’s eschewing evil;
With motions graceful as a bird’s in air;
Thou art, in sober truth, the veriest devil
That e’er clinched fingers in a captive’s hair!

That in thy breast there springs a poison fountain,
Deadlier than that where bathes the Upas-tree;
And in thy wrath a nursing cat-o’-mountain
Is calm as her babe's sleep compared with thee!

And underneath that face, like summer ocean’s,
Its lip as moveless, and its cheek as clear,
Slumbers a whirlwind of the heart’s emotions,
Love, hatred, pride, hope, sorrow—all save fear:

Love—for thy land, as if she were thy daughter,
Her pipe in peace, her tomahawk in wars;
Hatred—of missionaries and cold water;
Pride—in thy rifle-trophies and thy scars;

Hope—that thy wrongs may be, by the Great Spirit,
Remembered and revenged when thou art gone;
Sorrow—that none are left thee to inherit
Thy name, thy fame, thy passions, and thy throne!'''

# Function to analyze text features
def analyze_text(text):
    sentences = sent_tokenize(text)
    avg_len = sum(len(word_tokenize(sent)) for sent in sentences) / len(sentences) if sentences else 0
    
    pos_tags = pos_tag(word_tokenize(text))
    pos_freq = Counter(tag for word, tag in pos_tags)
    
    punctuation = Counter(char for char in text if char in '.,;!?')
    
    return avg_len, pos_freq, punctuation

# Analyze actual author's corpus
author_avg_lengths = []
author_pos_freqs = Counter()
author_punctuations = Counter()

for text in actual_author_texts:
    avg_len, pos_freq, punctuation = analyze_text(text)
    author_avg_lengths.append(avg_len)
    author_pos_freqs.update(pos_freq)
    author_punctuations.update(punctuation)

# Average statistics for the actual author
author_avg_length = statistics.mean(author_avg_lengths) if author_avg_lengths else 0
author_pos_freqs_normalized = {k: v / len(actual_author_texts) for k, v in author_pos_freqs.items()}
author_punctuations_normalized = {k: v / len(actual_author_texts) for k, v in author_punctuations.items()}

# Analyze the first disputed text
disputed_avg_length_1, disputed_pos_freqs_1, disputed_punctuations_1 = analyze_text(disputed_text_1)

# Analyze the second disputed text
disputed_avg_length_2, disputed_pos_freqs_2, disputed_punctuations_2 = analyze_text(disputed_text_2)

# Analyze the third disputed text
disputed_avg_length_3, disputed_pos_freqs_3, disputed_punctuations_3 = analyze_text(disputed_text_3)

# Compare sentence lengths
print(f"Average Sentence Length - Actual Author: {author_avg_length}")
print(f"Disputed Text 1: {disputed_avg_length_1}, Disputed Text 2: {disputed_avg_length_2}, Disputed Text 3: {disputed_avg_length_3}")

# Compare POS frequencies
print(f"POS Frequency - Actual Author: {author_pos_freqs_normalized}")
print(f"Disputed Text 1: {disputed_pos_freqs_1}")
print(f"Disputed Text 2: {disputed_pos_freqs_2}")
print(f"Disputed Text 3: {disputed_pos_freqs_3}")

# Compare punctuation usage
print(f"Punctuation - Actual Author: {author_punctuations_normalized}")
print(f"Punctuation - Disputed Text 1: {disputed_punctuations_1}")
print(f"Punctuation - Disputed Text 2: {disputed_punctuations_2}")
print(f"Punctuation - Disputed Text 3: {disputed_punctuations_3}")

